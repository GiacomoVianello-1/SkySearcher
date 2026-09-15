import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import numpy as np
import threading
import json

from std_srvs.srv import Trigger
from shapely.geometry import Polygon, MultiPolygon, Point as ShapelyPoint
from tf2_ros import Buffer, TransformListener

from .camera import Camera

from interfaces.srv import UpdatePosterior, AcquireObservation, VisualServoingVerify, GenerateWaypoints, EvaluateWaypoints, ResetBelief
from interfaces.msg import DetectionPositions

from .pilot import ManualPilotMonitor, NavigationResult

from sensor_msgs.msg import CameraInfo, Image as ImageMsg
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from interfaces.action import StartMission


class ExplorationPlannerNode(Node):
    def __init__(self):
        super().__init__("exploration_planner_node")

        # --- PARAMETERS ---

        # Topics and frames
        self.declare_parameter("camera_frame",      "Drone1/bottom_center_optical")
        self.declare_parameter("camera_info_topic", "/airsim_node/Drone1/bottom_center_Scene/camera_info")
        self.declare_parameter("odom_topic",        "/airsim_node/Drone1/odom_local")
        self.declare_parameter("image_topic",       "/airsim_node/Drone1/bottom_center_Scene/image")
        self.declare_parameter("odom_frame",        "Drone1")
        self.declare_parameter("world_frame",       "world")

        # Mission config
        self.declare_parameter("instruction", "")

        # Geometry
        self.declare_parameter("h_ground", 0.0)              # world z of ground plane
        self.declare_parameter("flight_altitude", 10.0)      # ENU (z is up)
        self.declare_parameter("max_projection_dist", 30.0)  # Distance to project rays for footprint (to avoid infinite projections when looking at the horizon)

        # Active domain
        self.declare_parameter("active_domain_radius", 24.0) # corona/bootstrap radius (m)

        # Posterior field evaluation
        self.declare_parameter("w_semantic", 1.0)          # weight factor to balance semantic IG vs geometric IG
        self.declare_parameter("w_reobs", 1.0)             # weight factor for yaw diversity re-observation IG

        # Candidate generation
        self.declare_parameter("frontier_spacing", 10.0)     # m between frontier samples

        # Cost traversal penalty (1/m) for candidate waypoints. J(w) = IG(w) - gamma * dist(current_pose, w).
        self.declare_parameter("gamma", 1.0)     # (1/m)

        # Movement
        self.declare_parameter("move_velocity", 2.0)         # assumed pilot cruise speed, only used to size timeouts
        self.declare_parameter("max_steps", 20)
        self.declare_parameter("waypoint_timeout_sec", 30.0)

        # Manual pilot: how close the pilot must get before the next iteration
        self.declare_parameter("arrival_position_tolerance", 1.00)  # m, XY
        self.declare_parameter("arrival_altitude_tolerance", 0.50)  # m, Z
        self.declare_parameter("arrival_yaw_tolerance_deg", 20.0)   # deg, 180 disables the yaw check
        self.declare_parameter("arrival_hold_time_sec", 1.0)        # must stay in tolerance this long
        self.declare_parameter("takeoff_timeout_sec", 120.0)

        # Debug & Modes
        self.declare_parameter("debug_planner", False)
        self.declare_parameter("marker_scale", 1.0)

        # Bounds (ENU world frame, same as planner coordinates)
        self.declare_parameter('bounds_x_min', -999999.0)
        self.declare_parameter('bounds_x_max',  999999.0)
        self.declare_parameter('bounds_y_min', -999999.0)
        self.declare_parameter('bounds_y_max',  999999.0)        

        # --- Resolve params ---
        self.camera_frame   = self.get_parameter("camera_frame").value
        self.odom_topic     = self.get_parameter("odom_topic").value
        camera_info_topic   = self.get_parameter("camera_info_topic").value
        self.image_topic    = str(self.get_parameter("image_topic").value)
        self.world_frame  = str(self.get_parameter('world_frame').value)
        self.odom_frame   = str(self.get_parameter('odom_frame').value)

        self.instruction    = str(self.get_parameter("instruction").value)
        self.primary_object = self.instruction
        
        self.h_ground           = float(self.get_parameter("h_ground").value)
        self.flight_altitude    = float(self.get_parameter("flight_altitude").value)
        self.max_projection_dist    = float(self.get_parameter("max_projection_dist").value)
        self.active_domain_radius   = float(self.get_parameter("active_domain_radius").value)
        self.marker_scale = float(self.get_parameter("marker_scale").value)
        self.w_semantic             = float(self.get_parameter("w_semantic").value)
        self.w_reobs                = float(self.get_parameter("w_reobs").value)

        self.frontier_spacing  = float(self.get_parameter("frontier_spacing").value)
        self.gamma             = float(self.get_parameter("gamma").value)

        self.move_velocity        = float(self.get_parameter("move_velocity").value)
        self.max_steps            = int(self.get_parameter("max_steps").value)
        self.waypoint_timeout_sec = float(self.get_parameter("waypoint_timeout_sec").value)

        self.arrival_pos_tol     = float(self.get_parameter("arrival_position_tolerance").value)
        self.arrival_alt_tol     = float(self.get_parameter("arrival_altitude_tolerance").value)
        self.arrival_yaw_tol_deg = float(self.get_parameter("arrival_yaw_tolerance_deg").value)
        self.arrival_hold_sec    = float(self.get_parameter("arrival_hold_time_sec").value)
        self.takeoff_timeout_sec = float(self.get_parameter("takeoff_timeout_sec").value)

        self.bounds_x_min = float(self.get_parameter('bounds_x_min').value)
        self.bounds_x_max = float(self.get_parameter('bounds_x_max').value)
        self.bounds_y_min = float(self.get_parameter('bounds_y_min').value)
        self.bounds_y_max = float(self.get_parameter('bounds_y_max').value)
        self._debug       = bool(self.get_parameter("debug_planner").value)

        # --- STATE ---
        self.camera = None
        self.current_pose = None  # (x, y, z, yaw) of the body in the world frame — not the camera optical pose
        self.coverage: Polygon | MultiPolygon | None = None
        self.map_polygon = Polygon([
            (self.bounds_x_min, self.bounds_y_min),
            (self.bounds_x_max, self.bounds_y_min),
            (self.bounds_x_max, self.bounds_y_max),
            (self.bounds_x_min, self.bounds_y_max),
        ])

        self.mission_active = False
        self.step_count = 0
        self._mission_distance_m = 0.0
        self._mission_last_feedback_xy: np.ndarray | None = None
        self._mission_stop_requested = False

        self.current_yaw = 0.0
        self._target_found: bool = False
        self._last_target_detected = False # To track changes in target detection state 
        self._failed_verifications: list[tuple[float, float]] = [] # Blacklist of failed verification locations (target_x, target_y)

        self._detection_mus: list[np.ndarray] = []   # List of arrays (2,)
        self._detection_radii: list[float] = []      # List of floats (sigma values from the VLM)
        self._detection_scale_x: list[float] = []    # List of floats (scale_x values from the VLM, i.e. ellipse major axis scaling)
        self._detection_scale_y: list[float] = []    # List of floats (scale_y values from the VLM, i.e. ellipse minor axis scaling)
        self._detection_yaw: list[float] = []        # List of floats (yaw values from the VLM, in degrees)

        self._detection_lock = threading.Lock()
        self._mission_admission_lock = threading.Lock()   # test-and-set of mission_active on goal acceptance

        # --- ROS INTERFACES ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        cb_group = ReentrantCallbackGroup()

        # --- SUBSCRIPTIONS ---
        self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(Bool, "/target_detected", self.target_detected_callback, 10)
        self.create_subscription(DetectionPositions, "/detection_positions", self._detection_positions_callback, 1)

        # --- SERVICE CLIENTS ---
        self.acquire_client             = self.create_client(AcquireObservation, "/acquire_observation", callback_group=cb_group)
        self.reset_coverage_client      = self.create_client(Trigger, "/reset_coverage", callback_group=cb_group)
        self.reset_belief_client          = self.create_client(ResetBelief, "/reset_semantic_belief", callback_group=cb_group)
        self.reset_coverage_belief_client = self.create_client(ResetBelief, "/reset_coverage_belief", callback_group=cb_group)
        self.update_posterior_client    = self.create_client(UpdatePosterior, "/update_posterior", callback_group=cb_group)
        self.generate_waypoints_client  = self.create_client(GenerateWaypoints, "/generate_waypoints", callback_group=cb_group)
        self.evaluate_waypoints_client  = self.create_client(EvaluateWaypoints, "/evaluate_waypoints", callback_group=cb_group)
        self.visual_servoing_client     = self.create_client(VisualServoingVerify, '/visual_servoing_verify', callback_group=cb_group)

        # The drone is flown by a human: the planner waits for arrival instead of commanding.
        self.pilot = ManualPilotMonitor(
            self, self.tf_buffer, self.world_frame, self.odom_frame,
            position_tolerance=self.arrival_pos_tol,
            altitude_tolerance=self.arrival_alt_tol,
            yaw_tolerance_deg=self.arrival_yaw_tol_deg,
            hold_time_sec=self.arrival_hold_sec,
            h_ground=self.h_ground,
            abort_fn=lambda: self._mission_stop_requested or not self.mission_active,
        )
 
        # Mission control action / stop service
        self.start_action = ActionServer(
            self,
            StartMission,
            "/start_mission",
            execute_callback=self.start_mission_execute_callback,
            goal_callback=self.start_mission_goal_callback,
            cancel_callback=self.start_mission_cancel_callback,
            callback_group=cb_group,
        )
        self.stop_srv   = self.create_service(Trigger, "/stop_mission", self.stop_mission_callback, callback_group=cb_group)

        # --- PUBLISHERS ---
        self.candidates_pub = self.create_publisher(MarkerArray, "/exploration_candidates", 10)
        self.chosen_pub     = self.create_publisher(Marker, "/exploration_chosen_waypoint", 10)
        self.domain_pub     = self.create_publisher(Marker, "/exploration_active_domain", 10)

        self._is_collision = False

        self.get_logger().info("Planner Node started")


    # --- CALLBACKS FOR STATE UPDATES ---

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera is None:
            self.camera = Camera(
                h=msg.height, w=msg.width,
                fx=msg.k[0], fy=msg.k[4],
                cx=msg.k[2], cy=msg.k[5],
                depth=self.max_projection_dist,
            )
            self.get_logger().info("Camera intrinsics loaded")

    def odom_callback(self, msg: Odometry):
        state = self.pilot.current_state()
        if state is not None:
            self.current_pose = state
            self.current_yaw = state[3]

    def target_detected_callback(self, msg: Bool):
        self._last_target_detected = msg.data

    def _detection_positions_callback(self, msg: DetectionPositions):
        mus = [np.array([p.x, p.y]) for p in msg.positions]
        radii = list(msg.radii)
        scale_x = list(msg.scale_x)
        scale_y = list(msg.scale_y)
        yaw = list(msg.yaw)
        with self._detection_lock:
            self._detection_mus = mus
            self._detection_radii = radii
            self._detection_scale_x = scale_x
            self._detection_scale_y = scale_y
            self._detection_yaw = yaw

    # --- MISSION CONTROL ---
    
    def _reset_coverage_map(self) -> bool:
        """Asks coverage_map_node to wipe its state before starting a new mission."""
        if not self.reset_coverage_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/reset_coverage service unavailable")
            return False
        result = self.reset_coverage_client.call(Trigger.Request())
        if result is None or not result.success:
            self.get_logger().error("Coverage map reset failed")
            return False
        self.get_logger().info("Coverage map reset OK.")
        return True


    def _grab_image(self, timeout_sec: float = 5.0) -> ImageMsg | None:
        """Block until one Image message arrives on the camera topic."""
        result = [None]
        event  = threading.Event()

        def _cb(msg: ImageMsg):
            result[0] = msg
            event.set()

        # Temporarily subscribe just to get a single frame
        cb_group = ReentrantCallbackGroup()
        sub = self.create_subscription(ImageMsg, self.image_topic, _cb, 1, callback_group=cb_group)
        event.wait(timeout=timeout_sec)
        self.destroy_subscription(sub)

        if result[0] is None:
            self.get_logger().warn(f"Timeout waiting for image on {self.image_topic}")
        return result[0]

    def start_mission_goal_callback(self, goal_request):
        full_inst = str(getattr(goal_request, 'full_instruction', '')).strip()
        if not full_inst:
            return GoalResponse.REJECT
        with self._mission_admission_lock:
            if self.mission_active:
                return GoalResponse.REJECT
            self.mission_active = True
        return GoalResponse.ACCEPT

    def start_mission_cancel_callback(self, goal_handle):
        self._mission_stop_requested = True
        self.mission_active = False
        return CancelResponse.ACCEPT

    def _reset_nodes_and_bounds(self, request):
        """Helper to invoke reset services on semantic_belief_tracker and coverage_map_node."""
        primary_obj = str(getattr(request, 'primary_object', '')).strip()
        full_inst = str(getattr(request, 'full_instruction', primary_obj)).strip()
        if not primary_obj:
            primary_obj = full_inst
        surroundings_json = getattr(request, 'surroundings_json', '')
        sigmas_json = getattr(request, 'sigmas_json', '')
        d_stars_json = getattr(request, 'd_stars_json', '')
        map_name = getattr(request, 'map_name', '')

        grid_x_min = float(getattr(request, 'grid_x_min', self.bounds_x_min))
        grid_x_max = float(getattr(request, 'grid_x_max', self.bounds_x_max))
        grid_y_min = float(getattr(request, 'grid_y_min', self.bounds_y_min))
        grid_y_max = float(getattr(request, 'grid_y_max', self.bounds_y_max))

        if grid_x_max > grid_x_min:
            self.bounds_x_min = grid_x_min
            self.bounds_x_max = grid_x_max
            self.bounds_y_min = grid_y_min
            self.bounds_y_max = grid_y_max
            self.map_polygon = Polygon([
                (self.bounds_x_min, self.bounds_y_min),
                (self.bounds_x_max, self.bounds_y_min),
                (self.bounds_x_max, self.bounds_y_max),
                (self.bounds_x_min, self.bounds_y_max),
            ])

        reset_req = ResetBelief.Request()
        reset_req.primary_object = primary_obj
        reset_req.object_description = str(getattr(request, 'object_description', '')).strip()
        reset_req.full_instruction = full_inst
        reset_req.surroundings_json = surroundings_json
        reset_req.sigmas_json = sigmas_json
        reset_req.d_stars_json = d_stars_json
        reset_req.map_name = map_name
        reset_req.grid_x_min = self.bounds_x_min
        reset_req.grid_x_max = self.bounds_x_max
        reset_req.grid_y_min = self.bounds_y_min
        reset_req.grid_y_max = self.bounds_y_max

        # Call Reset Belief tracker
        if self.reset_belief_client.wait_for_service(timeout_sec=2.0):
            try:
                self.reset_belief_client.call(reset_req)
                self.get_logger().info(f"[Planner] Reset semantic belief field for primary object '{primary_obj}'.")
            except Exception as e:
                self.get_logger().warn(f"[Planner] Failed calling reset_semantic_belief: {e}")

        # Call Reset Coverage Map
        if self.reset_coverage_belief_client.wait_for_service(timeout_sec=2.0):
            try:
                self.reset_coverage_belief_client.call(reset_req)
                self.get_logger().info("[Planner] Reset coverage belief field.")
            except Exception as e:
                self.get_logger().warn(f"[Planner] Failed calling reset_coverage_belief: {e}")
        else:
            self._reset_coverage_map()

    def start_mission_execute_callback(self, goal_handle):
        result = StartMission.Result()
        primary_obj = str(getattr(goal_handle.request, 'primary_object', '')).strip()
        full_inst = str(getattr(goal_handle.request, 'full_instruction', primary_obj)).strip()
        if not primary_obj:
            primary_obj = full_inst

        if self.camera is None or self.current_pose is None:
            result.success = False
            result.target_found = False
            result.steps = 0
            result.distance_travelled = 0.0
            result.message = "Camera or pose not yet available"
            self.mission_active = False
            goal_handle.abort()
            return result

        obj_desc = str(getattr(goal_handle.request, 'object_description', '')).strip()

        self.primary_object = primary_obj
        self.object_description = obj_desc
        self.full_instruction = full_inst
        self.instruction = primary_obj
        self.mission_active = True
        self._mission_stop_requested = False
        self.step_count = 0
        self.coverage = None
        self._target_found = False
        self._failed_verifications = []
        self._mission_distance_m = 0.0
        self._mission_last_feedback_xy = self._current_xy_from_pose()
        self._vlm_inference_time = 0.0

        self._is_collision = False

        # Perform full episode reset across nodes
        self._reset_nodes_and_bounds(goal_handle.request)

        mission_outcome = "aborted"
        try:
            if not self.wait_for_flight_altitude():
                self.get_logger().error("Takeoff failed, mission aborted")
                mission_outcome = "aborted"
                return self._finalize_start_mission_result(goal_handle, result, mission_outcome)

            self._target_found = False
            time.sleep(2.0)
            mission_outcome = self.mission_loop(goal_handle)
        finally:
            self.mission_active = False

        return self._finalize_start_mission_result(goal_handle, result, mission_outcome)

    def _finalize_start_mission_result(self, goal_handle, result, mission_outcome: str):
        result.steps = int(self.step_count)
        result.distance_travelled = float(self._mission_distance_m)
        vlm_time = float(round(getattr(self, '_vlm_inference_time', 0.0), 2))
        if hasattr(result, "vlm_inference_time"):
            result.vlm_inference_time = vlm_time
        if hasattr(result, "is_collision"):
            result.is_collision = bool(getattr(self, '_is_collision', False))

        if mission_outcome == "target_found":
            result.success = True
            result.target_found = True
            result.message = "Mission completed successfully: target verified."
            goal_handle.succeed()
        elif mission_outcome == "collision":
            result.success = False
            result.target_found = False
            if hasattr(result, "is_collision"):
                result.is_collision = True
            obj_str = getattr(self, '_collided_object', '')
            result.message = f"Mission aborted due to collision with '{obj_str if obj_str else 'obstacle'}'."
            goal_handle.abort()
        elif mission_outcome in {"canceled", "stopped"}:
            result.success = False
            result.target_found = False
            result.message = "Mission canceled before completion."
            goal_handle.canceled()
        else:
            result.success = False
            result.target_found = False
            if mission_outcome == "max_steps":
                result.message = "Mission finished without verifying the target (max steps reached)."
            else:
                result.message = "Mission aborted before completion."
            goal_handle.abort()
        return result

    def stop_mission_callback(self, request, response):
        self.mission_active = False
        self._mission_stop_requested = True
        response.success = True
        response.message = "Mission stop requested"
        return response

    # --- MAIN LOOP ---

    def mission_loop(self, goal_handle=None):
        """
        Main mission loop:
        - Acquire current observation and update coverage polygon(s)
        - Update the posterior field with the new observation
        - Check stopping criteria (target detection + confidence threshold)
        - Generate and evaluate candidate waypoints, choose the best one
        - Send move command to the chosen waypoint and repeat until stopping criteria are met or max steps are reached.
        """
        mission_outcome = "max_steps"
        while self.mission_active and self.step_count < self.max_steps:

            if self._mission_stop_requested:
                mission_outcome = "stopped"
                break
            if goal_handle is not None and goal_handle.is_cancel_requested:
                mission_outcome = "canceled"
                break

            self.step_count += 1
            self.get_logger().info(f" --- [{self.step_count}/{self.max_steps}] ---")

            # Acquire the current geometric footprint and update the coverage polygon(s)
            self.acquire_observation()
            
            # Update the posterior with the new observation
            self.update_posterior()

            if self._last_target_detected:
                if self._verify_target():
                    self._target_found = True
                    self._update_mission_distance_from_current_pose()
                    if goal_handle is not None:
                        self._publish_mission_feedback(goal_handle)
                    mission_outcome = "target_found"
                    break
                self.get_logger().info("[verify] False positive — resuming exploration.")

            # Generate and evaluate candidates
            best_w, _, best_yaw = self.choose_next_waypoint()
            if best_w is None:
                self.get_logger().warn("No valid candidate found, aborting mission.")
                mission_outcome = "aborted"
                break
            
            # --- Move Logic ---
            nav_res = self.send_move_command(best_w[0], best_w[1], best_yaw)

            if nav_res == NavigationResult.CANCELED:
                mission_outcome = "stopped"
                break

            if nav_res != NavigationResult.SUCCESS:
                self.get_logger().warn(
                    f"⚠️ Waypoint not reached (result: {nav_res.value}). "
                    f"Acquiring footprint at the current pose and continuing mission."
                )
                self._update_mission_distance_from_current_pose()
                if goal_handle is not None:
                    self._publish_mission_feedback(goal_handle)
                time.sleep(2.0)
                self.acquire_observation()
                self.update_posterior()
                continue

            self._update_mission_distance_from_current_pose()
            if goal_handle is not None:
                self._publish_mission_feedback(goal_handle)
            time.sleep(2.0)
            # ------

        if self._target_found:
            mission_outcome = "target_found"
        elif self._mission_stop_requested:
            mission_outcome = "stopped"
        elif goal_handle is not None and goal_handle.is_cancel_requested:
            mission_outcome = "canceled"
        elif self.step_count >= self.max_steps:
            mission_outcome = "max_steps"
        elif mission_outcome == "max_steps" and not self.mission_active:
            mission_outcome = "aborted"

        self.mission_active = False

        return mission_outcome

    def _current_xy_from_pose(self) -> np.ndarray | None:
        if self.current_pose is None:
            return None
        return np.array([float(self.current_pose[0]), float(self.current_pose[1])], dtype=float)

    def _update_mission_distance_from_current_pose(self):
        current_xy = self._current_xy_from_pose()
        if current_xy is None:
            return
        if self._mission_last_feedback_xy is None:
            self._mission_last_feedback_xy = current_xy
            return
        self._mission_distance_m += float(np.linalg.norm(current_xy - self._mission_last_feedback_xy))
        self._mission_last_feedback_xy = current_xy

    def _publish_mission_feedback(self, goal_handle):
        feedback = StartMission.Feedback()
        feedback.iteration = int(self.step_count)
        if hasattr(feedback, "is_collision"):
            feedback.is_collision = bool(getattr(self, '_is_collision', False))
        state = self.pilot.current_state()
        if state is not None:
            self.current_pose = state
            self.current_yaw = state[3]
        current_xy = self._current_xy_from_pose()
        point = Point()
        if current_xy is not None and self.current_pose is not None:
            point.x = float(current_xy[0])
            point.y = float(current_xy[1])
            point.z = float(self.current_pose[2])
        feedback.position = point
        feedback.distance_travelled = float(self._mission_distance_m)
        goal_handle.publish_feedback(feedback)

    # --- STEP COMPONENTS ---

    def acquire_observation(self) -> bool:
        """
        Calls `acquire_observation` service (exposed by coverage_map_node) to get the current coverage polygon(s)
        and update the current geometric footprint that the camera saw (self.coverage).
        """
        if not self.acquire_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/acquire_observation service unavailable — skipping")
            return False

        req = AcquireObservation.Request()
        result = self.acquire_client.call(req)
        if result is None or not result.success:
            self.get_logger().error(f"Acquire failed: {result.message if result else 'no response'}")
            return False

        try:
            parts = []
            for cp in result.polygons:
                if len(cp.xs) >= 3:
                    p = Polygon(list(zip(cp.xs, cp.ys)))
                    if p.is_valid and p.area > 0.0:
                        parts.append(p)
            if parts:
                self.coverage = parts[0] if len(parts) == 1 else MultiPolygon(parts)
            else:
                self.get_logger().error("Acquire: no valid polygons in response — coverage unchanged")
        except Exception as e:
            self.get_logger().error(f"Error in coverage parsing: {e}")
            return False

        # result.message contains both the total area and the number of connected components
        self.get_logger().info(f"[Geometric Map Updated]: {result.message}")
        return True
    
    def update_posterior(self) -> str:
        """
        Calls the `update_posterior` service that asks the semantic_belief_tracker to update the 
        posterior field F based on the new observation and the current coverage (ROI).
        Returns result.message ("TARGET_FOUND" or "OK").
        """
        if not self.update_posterior_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("/update_posterior service unavailable — skipping")
            return ""

        req = UpdatePosterior.Request()

        # ROI: bbox of current coverage + margin, so the tracker only needs to publish the relevant slice of the field F
        if self.coverage is not None and not self.coverage.is_empty:
            xmin, ymin, xmax, ymax = self.coverage.bounds
            margin = self.active_domain_radius
            req.roi_x_min = xmin - margin
            req.roi_x_max = xmax + margin
            req.roi_y_min = ymin - margin
            req.roi_y_max = ymax + margin

        t0 = time.time()
        result = self.update_posterior_client.call(req)
        self._vlm_inference_time += (time.time() - t0)
        if result is None or not result.success:
            msg = result.message if result else "no response"
            self.get_logger().error(f"Failed to update posterior: {msg}")
            return ""

        #self.get_logger().info(f"Posterior update: {result.message}")
        return result.message

    def choose_next_waypoint(self):
        """Returns (best_xy, best_ig, best_yaw) or (None, 0, 0)."""
        if self.current_pose is None:
            return None, 0.0, 0.0

        cx, cy = float(self.current_pose[0]), float(self.current_pose[1])

        # 1. Call /generate_waypoints service
        req_gen = GenerateWaypoints.Request()
        req_gen.current_x = cx
        req_gen.current_y = cy
        req_gen.current_yaw = float(self.current_yaw)
        req_gen.frontier_spacing = float(self.frontier_spacing)
        req_gen.standoff_dist = 8.0

        res_gen = self.generate_waypoints_client.call(req_gen)
        if res_gen is None or not res_gen.success or not res_gen.candidates:
            self.get_logger().warn("No candidates generated by /generate_waypoints")
            return None, 0.0, 0.0

        candidates = res_gen.candidates
        cand_tuples = [(c.x, c.y, c.theta) for c in candidates]
        self.publish_candidates(cand_tuples)  # RViz visualization

        # 2. Call /evaluate_waypoints service
        req_eval = EvaluateWaypoints.Request()
        req_eval.candidates = candidates
        req_eval.current_x = cx
        req_eval.current_y = cy
        req_eval.gamma = float(self.gamma)
        req_eval.w_semantic = float(self.w_semantic)
        req_eval.w_reobs = float(self.w_reobs)

        res_eval = self.evaluate_waypoints_client.call(req_eval)
        if res_eval is None or not res_eval.success or res_eval.best_index < 0:
            self.get_logger().warn("Waypoints evaluation failed")
            return None, 0.0, 0.0

        best_idx = res_eval.best_index
        best_wp = res_eval.best_waypoint
        best_score = res_eval.scores[best_idx] if best_idx < len(res_eval.scores) else 0.0

        best = ((best_wp.x, best_wp.y), best_score, best_wp.theta)
        self.publish_best_candidate(best)  # RViz visualization

        dist = np.hypot(best_wp.x - cx, best_wp.y - cy)
        best_ig_geom  = res_eval.ig_geometric[best_idx]
        best_ig_sem   = res_eval.ig_semantic[best_idx]
        best_ig_reobs = res_eval.ig_reobservation[best_idx]
        penalty = float(self.gamma) * dist
        self.get_logger().info(
            f"[w*] pos=({best_wp.x:.1f},{best_wp.y:.1f}), yaw={np.degrees(best_wp.theta):.1f}° | "
            f"dist={dist:.1f}m | IG_geom={best_ig_geom:.5f} | IG_sem={best_ig_sem:.5f} | IG_reobs={best_ig_reobs:.5f} | penalty={penalty:.5f} | J={best_score:.5f}"
        )

        return (best_wp.x, best_wp.y), best_score, best_wp.theta

    # -- MOVE COMMANDS ---

    def _within_bounds(self, x: float, y: float) -> bool:
        return (self.bounds_x_min <= x <= self.bounds_x_max and self.bounds_y_min <= y <= self.bounds_y_max)

    # --- PILOT COMMANDS ---
    # The drone is flown by a human. These three methods are the only place
    # that knows it: swapping in AeroStack2 means reimplementing them.

    def send_move_command(self, x, y, yaw, z=None, timeout_sec=None) -> NavigationResult:
        """Ask the pilot for a viewpoint and block until they hold it."""
        if not self._within_bounds(x, y):
            self.get_logger().warn(f'Waypoint ({x:.2f}, {y:.2f}) outside bounds 'f'x=[{self.bounds_x_min}, {self.bounds_x_max}] 'f'y=[{self.bounds_y_min}, {self.bounds_y_max}] — skipping.')
            return NavigationResult.OUT_OF_BOUNDS

        target_z = z if z is not None else self.flight_altitude
        if timeout_sec is None:
            state = self.pilot.current_state()
            if state is not None:
                dist = float(np.hypot(x - state[0], y - state[1]))
                expected_time = dist / max(self.move_velocity, 0.1)
                timeout_sec = max(self.waypoint_timeout_sec, expected_time + 15.0)
            else:
                timeout_sec = self.waypoint_timeout_sec

        return self.pilot.wait_until_at(x, y, target_z, yaw, timeout_sec, label="waypoint")

    def send_rotate_command(self, yaw_rad: float) -> bool:
        """Ask the pilot to yaw in place."""
        state = self.pilot.current_state()
        if state is None:
            self.get_logger().error("No pose available to hold position while yawing")
            return False
        cx, cy, cz, _ = state
        res = self.pilot.wait_until_at(cx, cy, cz, yaw_rad, timeout_sec=30.0, label="heading")
        return res == NavigationResult.SUCCESS

    def wait_for_flight_altitude(self) -> bool:
        """Wait for the pilot to take off and reach flight_altitude at the current XY."""
        state = self.pilot.current_state()
        if state is None:
            self.get_logger().error("No pose available for climb to flight altitude")
            return False

        cx, cy, cz, yaw_curr = state
        if cz > self.flight_altitude + self.arrival_alt_tol:
            self.get_logger().error(f"❗ The drone is already above the cruise altitude (z = {cz:.1f} m > {self.flight_altitude:.1f} m).")
            return False

        res = self.pilot.wait_until_at(cx, cy, self.flight_altitude, yaw_curr,
                                       timeout_sec=self.takeoff_timeout_sec, label="takeoff altitude")
        if res != NavigationResult.SUCCESS:
            self.get_logger().error(f"Climb to flight altitude failed ({res.value})")
            return False

        self.get_logger().info(f"Reached: {self.flight_altitude:.1f} m.")
        return True


    def _verify_target(self) -> bool:
        """
        Delegates target candidate verification to /visual_servoing_verify service.
        Returns True if confirmed, False otherwise.
        """
        # Save pre-descent pose to return to in case of false positive or verification failure
        pre_x, pre_y, pre_z, pre_yaw = (float(v) for v in self.current_pose)

        with self._detection_lock:
            mus    = list(self._detection_mus)
            radii  = list(self._detection_radii)
            sx_list = list(self._detection_scale_x)
            sy_list = list(self._detection_scale_y)

        target_indices = [i for i, r in enumerate(radii) if r == 0.0]
        if not target_indices:
            return False

        best_idx   = max(target_indices, key=lambda i: sx_list[i] * sy_list[i])
        mu         = mus[best_idx]
        target_x   = float(mu[0])
        target_y   = float(mu[1])

        # Ignore target candidate if it was already verified as a false positive
        for fx, fy in self._failed_verifications:
            d_prev = float(np.hypot(target_x - fx, target_y - fy))
            if d_prev < 5.0:
                self.get_logger().info(f"🚫 [verify] BLACKLIST HIT: Candidate at ({target_x:.1f}, {target_y:.1f}) is {d_prev:.1f}m from previously failed verification at ({fx:.1f}, {fy:.1f}). Skipping verification.")
                return False

        # Delegate verification to VisualServoingNode
        if not self.visual_servoing_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("[verify] /visual_servoing_verify service unavailable")
            return False

        self.get_logger().info(f"Starting Visual Servoing verification...")
        vs_req = VisualServoingVerify.Request()
        vs_req.target_x = target_x
        vs_req.target_y = target_y
        vs_req.pre_x = pre_x
        vs_req.pre_y = pre_y
        vs_req.pre_z = pre_z
        vs_req.pre_yaw = pre_yaw
        vs_req.target_query = self.primary_object if getattr(self, 'primary_object', '') else self.instruction
        vs_req.object_description = getattr(self, 'object_description', '')
        vs_req.max_adjust_steps = 6

        t0 = time.time()
        vs_res = self.visual_servoing_client.call(vs_req)
        self._vlm_inference_time += (time.time() - t0)
        if vs_res is not None and vs_res.verified:
            # Update current pose with the final close-range verified pose reported by the servoing node
            self.current_pose = (float(vs_res.final_x), float(vs_res.final_y),
                                 float(vs_res.final_z), float(vs_res.final_yaw))
            self.current_yaw = self.current_pose[3]

            self.get_logger().info(f"🎯 TARGET VERIFIED via Closed-Loop Visual Servoing at close range ({vs_res.final_x:.1f}, {vs_res.final_y:.1f}, z={vs_res.final_z:.1f}m)!")
            self.get_logger().info(f"├── 👟  Mission ended after {self.step_count} steps.")
            self.get_logger().info(f"├── 🗺️  Total area covered: {self.coverage.area if self.coverage else 0:.2f} m^2.")
            self.get_logger().info(f"└── 📏  Total distance travelled: {self._mission_distance_m:.1f} m.")
            return True

        self._failed_verifications.append((target_x, target_y))
        self.get_logger().info(f"❌ [verify] FALSE POSITIVE — Added ({target_x:.1f}, {target_y:.1f}) to blacklist (total blacklisted: {len(self._failed_verifications)}). Returning to pre-descent position.")
        self._return_to_pre_descent(pre_x, pre_y, pre_yaw)
        return False


    def _return_to_pre_descent(self, x: float, y: float, yaw: float):
        """Returns to the pre-descent position safely by climbing vertically first to avoid terrain collision."""
        self.get_logger().info(f"Returning to ({x:.1f},{y:.1f}) at {self.flight_altitude}m (climbing vertically first)...")
        # Phase 1: Ascend vertically to flight_altitude at current position to clear all terrain/obstacles
        if self.current_pose is not None:
            cx = float(self.current_pose[0])
            cy = float(self.current_pose[1])
            self.send_move_command(cx, cy, yaw, z=self.flight_altitude)

        # Phase 2: Move horizontally at flight_altitude to pre-descent position
        self.send_move_command(x, y, yaw, z=self.flight_altitude)

    # --- VISUALIZATION ---

    def publish_domain(self, domain):
        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "active_domain"
        m.id = 0
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale.x = 0.5 * self.marker_scale
        m.color.r, m.color.g, m.color.b, m.color.a = 0.4, 0.6, 1.0, 1.0
        z = self.h_ground + 0.10

        polys = (domain.geoms if isinstance(domain, MultiPolygon) else [domain])
        for p in polys:
            coords = list(p.exterior.coords)
            for i in range(len(coords) - 1):
                p1, p2 = coords[i], coords[i+1]
                m.points.append(Point(x=float(p1[0]), y=float(p1[1]), z=z))
                m.points.append(Point(x=float(p2[0]), y=float(p2[1]), z=z))
        self.domain_pub.publish(m)

    def publish_candidates(self, candidates):
        ma = MarkerArray()
        if not candidates:
            # Send a clear command if there are no candidates to display
            clear_marker = Marker()
            clear_marker.action = Marker.DELETEALL
            ma.markers.append(clear_marker)
            self.candidates_pub.publish(ma)
            return

        for i, (x, y, yaw) in enumerate(candidates):
            # --- Sphere (XY point) ---
            m_sphere = Marker()
            m_sphere.header.frame_id = self.world_frame
            m_sphere.ns = "pts"
            m_sphere.id = i
            m_sphere.type = Marker.SPHERE
            m_sphere.pose.position.x = float(x)
            m_sphere.pose.position.y = float(y)
            m_sphere.pose.position.z = self.h_ground + 0.2
            m_sphere.scale.x = m_sphere.scale.y = m_sphere.scale.z = 0.8 * self.marker_scale
            m_sphere.color.r, m_sphere.color.g, m_sphere.color.b, m_sphere.color.a = 0.8, 0.8, 0.8, 1.0
            ma.markers.append(m_sphere)

            # --- Arrow (Yaw) ---
            m_arrow = Marker()
            m_arrow.header.frame_id = self.world_frame
            m_arrow.ns = "yaw_arrows"
            m_arrow.id = i + 1000  # ID univoco
            m_arrow.type = Marker.ARROW
            
            # Positioning and orientation (via quaternions)
            m_arrow.pose.position.x = float(x)
            m_arrow.pose.position.y = float(y)
            m_arrow.pose.position.z = self.h_ground + 0.3
            
            # Converts yaw to quaternion (Z-axis rotation)
            m_arrow.pose.orientation.z = np.sin(yaw / 2.0)
            m_arrow.pose.orientation.w = np.cos(yaw / 2.0)
            
            # Size: x=length, y=width, z=height arrow
            m_arrow.scale.x, m_arrow.scale.y, m_arrow.scale.z = 2.0 * self.marker_scale, 0.4 * self.marker_scale, 0.4 * self.marker_scale
            m_arrow.color.r, m_arrow.color.g, m_arrow.color.b, m_arrow.color.a = 1.0, 1.0, 1.0, 1.0 # White for the normal
            ma.markers.append(m_arrow)

        self.candidates_pub.publish(ma)
    
    def publish_best_candidate(self, best):
        mc_clear = Marker()
        mc_clear.header.frame_id = self.world_frame
        mc_clear.header.stamp = self.get_clock().now().to_msg()
        mc_clear.ns = "chosen"
        mc_clear.action = Marker.DELETEALL
        self.chosen_pub.publish(mc_clear)

        if best is not None:
            w, ig, yaw = best
            mc = Marker()
            mc.header.frame_id = self.world_frame
            mc.header.stamp = self.get_clock().now().to_msg()
            mc.ns = "chosen"
            mc.id = 0
            mc.type = Marker.ARROW
            mc.action = Marker.ADD
            
            arrow_length = 10.0 * self.marker_scale
            start = Point(x=float(w[0]), y=float(w[1]), z=self.h_ground + 1.0)
            end = Point(
                x=float(w[0] + arrow_length * np.cos(yaw)),
                y=float(w[1] + arrow_length * np.sin(yaw)),
                z=self.h_ground + 1.0
            )
            
            mc.points = [start, end]
            mc.scale.x, mc.scale.y, mc.scale.z = 2.0 * self.marker_scale, 4.0 * self.marker_scale, 6.0 * self.marker_scale
            mc.color.r, mc.color.g, mc.color.b, mc.color.a = 1.0, 1.0, 0.0, 1.0
            self.chosen_pub.publish(mc)

# --- MAIN ---

def main(args=None):
    rclpy.init(args=args)
    node = ExplorationPlannerNode()
    executor = MultiThreadedExecutor() # MultiThreadedExecutor lets services and the mission loop run concurrently
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()