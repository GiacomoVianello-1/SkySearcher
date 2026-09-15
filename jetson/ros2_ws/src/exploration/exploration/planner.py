import time 
import json
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from cv_bridge import CvBridge
import cv2, os

import numpy as np
import threading

from std_srvs.srv import Trigger
from shapely.geometry import Polygon, MultiPolygon, Point as ShapelyPoint
from tf2_ros import Buffer, TransformListener

from .camera import Camera
from .utils import compute_ground_footprint, lookup_camera_pose, lookup_body_yaw, lookup_static_transform, build_candidate_pose

from interfaces.srv import UpdatePosterior, AcquireObservation, VerifyTarget

from interfaces.msg import PosteriorField as PosteriorFieldMsg, DetectionPositions
from .active_domain import compute_active_domain as active_domain_fn

from sensor_msgs.msg import CameraInfo, Image as ImageMsg
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from interfaces.action import StartMission

from shapely import prepare, contains_xy
from shapely.affinity import scale as shapely_scale, rotate as shapely_rotate

from as2_msgs.action import GoToWaypoint, Takeoff, Land
from as2_msgs.msg import YawMode
from rclpy.action import ActionClient
from scipy.spatial.transform import Rotation

class ExplorationPlannerNode(Node):
    def __init__(self):
        super().__init__("exploration_planner_node")

        # --- PARAMETERS ---

        # Topics and frames
        self.declare_parameter("camera_frame",      "camera_color_optical_frame")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("odom_topic",        "/vrpn_mocap/jetson_nx/pose")
        self.declare_parameter("image_topic",       "/camera/camera/color/image_raw")

        # Mission config
        self.declare_parameter("instruction",   "red rowing boat in the lake")
        self.declare_parameter("vlm_log_path",  "vlm_scale_clues_log.json")
        self.declare_parameter("max_steps", 25)

        # Geometry
        self.declare_parameter("h_ground",            0.0)   # World z of ground plane
        self.declare_parameter("flight_altitude",     10.0)  # ENU (z is up)
        self.declare_parameter("max_projection_dist", 30.0)  # Distance to project rays for footprint (to avoid infinite projections when looking at the horizon)

        # Active domain
        self.declare_parameter("active_domain_radius", 24.0) # Corona radius (m)

        # Posterior field
        self.declare_parameter("pi_0", 0.05)                 # uniform prior on D_t

        # Candidate generation
        self.declare_parameter("frontier_spacing", 20.0)      # m between frontier samples
        self.declare_parameter("gamma", 1.0)     # (1/m)

        # Debug
        self.declare_parameter("debug_planner", False) 

        # Bounds (ENU world frame, same as planner coordinates)
        self.declare_parameter('bounds_x_min', -999999.0)
        self.declare_parameter('bounds_x_max',  999999.0)
        self.declare_parameter('bounds_y_min', -999999.0)
        self.declare_parameter('bounds_y_max',  999999.0)

        self.declare_parameter("toy_run", True)  # True = skip real movements

        self.declare_parameter("save_images", False)
        self.declare_parameter("save_dir", "/root/ros2_ws/saved_images/indoor_run")

        # --- Resolve params ---
        self.camera_frame           = self.get_parameter("camera_frame").value
        self.odom_topic             = self.get_parameter("odom_topic").value
        camera_info_topic           = self.get_parameter("camera_info_topic").value
        self.image_topic            = str(self.get_parameter("image_topic").value)
        self.instruction            = str(self.get_parameter("instruction").value)
        self.vlm_log_path           = str(self.get_parameter("vlm_log_path").value)
        self.max_steps              = int(self.get_parameter("max_steps").value)
        self.h_ground               = float(self.get_parameter("h_ground").value)
        self.flight_altitude        = float(self.get_parameter("flight_altitude").value)
        self.max_projection_dist    = float(self.get_parameter("max_projection_dist").value)
        self.active_domain_radius   = float(self.get_parameter("active_domain_radius").value)
        self.pi_0                   = float(self.get_parameter("pi_0").value)
        self.frontier_spacing       = float(self.get_parameter("frontier_spacing").value)
        self.gamma                  = float(self.get_parameter("gamma").value)
        self.bounds_x_min           = float(self.get_parameter('bounds_x_min').value)
        self.bounds_x_max           = float(self.get_parameter('bounds_x_max').value)
        self.bounds_y_min           = float(self.get_parameter('bounds_y_min').value)
        self.bounds_y_max           = float(self.get_parameter('bounds_y_max').value)
        self._debug                 = bool(self.get_parameter("debug_planner").value)

        # --- STATE ---
        self.camera = None
        self.current_pose = None
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

        self.takeoff_altitude = self.flight_altitude
        self.T_body_to_opt = None 
        self.current_yaw = 0.0
        self._target_found: bool = False
        self._last_target_detected = False  # To track changes in target detection state 
        self._last_eval_breakdown = None    # For debugging IG evaluation: (roi_area, ig_geometric, ig_semantic, fp_area)

        self._detection_mus: list[np.ndarray] = []   # List of arrays (2,)
        self._detection_radii: list[float] = []      # List of floats (sigma values from the VLM)
        self._detection_scale_x: list[float] = []    # List of floats (scale_x values from the VLM, i.e. ellipse major axis scaling)
        self._detection_scale_y: list[float] = []    # List of floats (scale_y values from the VLM, i.e. ellipse minor axis scaling)
        self._detection_yaw: list[float] = []        # List of floats (yaw values from the VLM, in degrees)

        self._detection_lock = threading.Lock()

        # Posterior field state — kept in sync via /posterior_field subscription
        self._field_lock = threading.Lock()
        self._field_F: np.ndarray | None = None
        self._field_origin_x: float = 0.0
        self._field_origin_y: float = 0.0
        self._field_resolution: float = 0.0
        self._field_width: int = 0
        self._field_height: int = 0

        self._waypoint_confirmed = False # Confirm the observation and frustum acquisition via service (meant for manual mode)

        self._annotated_image = None
        self._annotated_image_lock = threading.Lock()

        # --- EVALUATION METRICS ---
        self._step_ig_geometric  = 0.0
        self._step_ig_semantic   = 0.0
        self._step_coverage_area = 0.0
        self._step_max_posterior = 0.0
        self._step_vlm_time      = 0.0
        self._step_target_est_x  = None
        self._step_target_est_y  = None

        # --- ROS INTERFACES ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        cb_group = ReentrantCallbackGroup()

        mocap_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # --- SUBSCRIPTIONS ---
        self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(PoseStamped, self.odom_topic, self.odom_callback, mocap_qos)
        self.create_subscription(Bool, "/target_detected", self.target_detected_callback, 10)
        self.create_subscription(DetectionPositions, "/detection_positions", self._detection_positions_callback, 1)
        self.create_subscription(PosteriorFieldMsg, "/posterior_field", self.posterior_field_callback, 1)
        self.create_subscription(ImageMsg, "/annotated_img", self._annotated_image_callback, 10)

        # --- SERVICE CLIENTS ---
        self.acquire_client          = self.create_client(AcquireObservation, "/acquire_observation", callback_group=cb_group)
        self.reset_coverage_client   = self.create_client(Trigger, "/reset_coverage", callback_group=cb_group)
        self.update_posterior_client = self.create_client(UpdatePosterior, "/update_posterior", callback_group=cb_group)
        self.verify_client           = self.create_client(VerifyTarget, '/verify_target', callback_group=cb_group)
        self.confirm_wp_srv          = self.create_service(Trigger, "/confirm_waypoint", self._confirm_waypoint_callback)

        # Aerostack2
        self.takeoff_client   = ActionClient(self, Takeoff,      '/drone0/TakeOffBehaviour',    callback_group=cb_group)
        self.gotowp_client    = ActionClient(self, GoToWaypoint, '/drone0/GoToWaypointBehaviour', callback_group=cb_group)
        self.land_client      = ActionClient(self, Land,         '/drone0/LandBehaviour',        callback_group=cb_group)
 
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
        self.domain_pub     = self.create_publisher(Marker,   "/exploration_active_domain", 10)

        self.get_logger().info("Planner Node started — send a StartMission goal to /start_mission")


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

    def odom_callback(self, msg: PoseStamped):
        if self.T_body_to_opt is None:
            T = lookup_static_transform(self.tf_buffer, "jetson_nx", self.camera_frame)
            if T is not None:
                self.T_body_to_opt = T

        yaw = lookup_body_yaw(self.tf_buffer, "world", "jetson_nx")
        if yaw is not None:
            self.current_yaw = yaw

        T_cam = lookup_camera_pose(self.tf_buffer, "world", self.camera_frame)
        if T_cam is not None:
            self.current_pose = T_cam

    def target_detected_callback(self, msg: Bool):
        self._last_target_detected = msg.data

    def posterior_field_callback(self, msg: PosteriorFieldMsg):
        """Cache the latest posterior field and its metadata."""
        F = np.array(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        with self._field_lock:
            self._field_F = F
            self._field_origin_x = msg.origin_x
            self._field_origin_y = msg.origin_y
            self._field_resolution = msg.resolution
            self._field_width = msg.width
            self._field_height = msg.height

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
    
    def _confirm_waypoint_callback(self, request, response):
        self._waypoint_confirmed = True
        response.success = True
        response.message = "Waypoint confirmed."
        return response

    def _annotated_image_callback(self, msg: ImageMsg):
        with self._annotated_image_lock:
            self._annotated_image = msg

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
        if self.mission_active:
            return GoalResponse.REJECT
        instruction = str(goal_request.instruction).strip()
        if not instruction:
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def start_mission_cancel_callback(self, goal_handle):
        self._mission_stop_requested = True
        self.mission_active = False
        return CancelResponse.ACCEPT

    def start_mission_execute_callback(self, goal_handle):
        result = StartMission.Result()
        instruction = str(goal_handle.request.instruction).strip()

        if self.camera is None or self.current_pose is None:
            result.success = False
            result.target_found = False
            result.steps = 0
            result.distance_travelled = 0.0
            result.message = "Camera or pose not yet available"
            goal_handle.abort()
            return result

        self.instruction = instruction
        self._save_dir = None
        self.mission_active = True
        self._mission_stop_requested = False
        self.step_count = 0
        self.coverage = None
        self._target_found = False
        self._mission_distance_m = 0.0
        self._mission_last_feedback_xy = self._current_xy_from_pose()

        mission_outcome = "aborted"
        try:
            if not self._reset_coverage_map():
                self.get_logger().warn("Coverage map reset failed — proceeding anyway (possible phantom polygons).")
            if not self.arm_and_takeoff():
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

        if mission_outcome == "target_found":
            result.success = True
            result.target_found = True
            result.message = "Mission completed successfully: target verified."
            goal_handle.succeed()
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

    def _wait_until_at_waypoint(self, target_x: float, target_y: float) -> bool:
        if not self.get_parameter("toy_run").value:
            # Aerostack2: GoToWaypoint is already blocking --> need for wait loop
            return True

        self._waypoint_confirmed = False
        timeout = 120.0  # seconds — in case something goes wrong, don't wait indefinitely
        t_start = self.get_clock().now().nanoseconds * 1e-9

        self.get_logger().info(
            f"[manual] Carry drone to ({target_x:.2f}, {target_y:.2f}). "
            f"When ready: ros2 service call /confirm_waypoint std_srvs/srv/Trigger '{{}}'"
        )

        while rclpy.ok():
            if self._mission_stop_requested:
                return False
            if self._waypoint_confirmed:
                self.get_logger().info("[manual] Waypoint confirmed by operator.")
                return True
            elapsed = self.get_clock().now().nanoseconds * 1e-9 - t_start
            if elapsed > timeout:
                self.get_logger().warn(f"[manual] Timeout ({timeout}s) — proceeding anyway.")
                return False
            time.sleep(0.2)

    def _camera_forward_offset(self) -> float:
        if self.camera is None or self.current_pose is None:
            return 0.0

        h = float(self.current_pose[2, 3]) - self.h_ground
        if h <= 0:
            return 0.0

        fp = compute_ground_footprint(self.current_pose, self.camera, self.h_ground, self.max_projection_dist)
        if not fp.is_valid or fp.is_empty:
            return 0.0

        drone_xy = ShapelyPoint(self.current_pose[0, 3], self.current_pose[1, 3])
        
        # Distance from the drone to the closest edge of the projected frustum footprint
        return float(drone_xy.distance(fp.exterior))

    # --- MAIN LOOP ---

    def mission_loop(self, goal_handle=None):
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

            if self.get_parameter("save_images").value:
                self._save_annotated_image(self.step_count)

            with self._field_lock:
                current_max_p = float(self._field_F.max()) if self._field_F is not None else 0.0
                F_copy = self._field_F.copy() if self._debug and self._field_F is not None else None

            # *** DEBUG *** 
            if self._debug and F_copy is not None:
                n_high = int((F_copy > 0.5).sum())
                idx = np.unravel_index(np.argmax(F_copy), F_copy.shape)
                x_max = self._field_origin_x + (idx[1] + 0.5) * self._field_resolution
                y_max = self._field_origin_y + (idx[0] + 0.5) * self._field_resolution
                self.get_logger().info(
                    f"[DBG field] max(F)={current_max_p:.3f} at ({x_max:.1f},{y_max:.1f}), "
                    f"cells>0.5: {n_high}"
                )
            # ******

            if self._last_target_detected:
                self.get_logger().info("[verify] Target detected — starting close-range verification...")
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
                break
            
            # --- Move Logic ---
            cx, cy = self.current_pose[0, 3], self.current_pose[1, 3]
            dx, dy = best_w[0] - cx, best_w[1] - cy
            d = np.hypot(dx, dy)
            self.get_logger().info(f"Distance to Waypoint: {d:.1f}m")

            ok = self.send_move_command(best_w[0], best_w[1], best_yaw)
            if ok:
                # Wait for reaching next waypoint
                self._wait_until_at_waypoint(best_w[0], best_w[1])

            if not ok:
                self.get_logger().error("Move command failed, aborting")
                mission_outcome = "aborted"
                break

            self._update_mission_distance_from_current_pose()
            if goal_handle is not None:
                self._publish_mission_feedback(goal_handle)
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

        self.get_logger().info(
            f"\n👟  Mission ended after {self.step_count} steps. \n"
            f"🗺️  Total area covered: {self.coverage.area if self.coverage else 0:.2f} m^2. \n"
            f"📏  Total distance travelled: {self._mission_distance_m:.1f} m."
        )
        return mission_outcome

    def _current_xy_from_pose(self) -> np.ndarray | None:
        if self.current_pose is None:
            return None
        return np.array([float(self.current_pose[0, 3]), float(self.current_pose[1, 3])], dtype=float)

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
        feedback.iteration          = int(self.step_count)
        current_xy = self._current_xy_from_pose()
        point = Point()
        if current_xy is not None:
            point.x = float(current_xy[0])
            point.y = float(current_xy[1])
            point.z = float(self.current_pose[2, 3])
        feedback.position               = point
        if self.current_pose is not None:
            R = self.current_pose[:3, :3]
            from scipy.spatial.transform import Rotation
            q = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
            feedback.orientation.x = float(q[0])
            feedback.orientation.y = float(q[1])
            feedback.orientation.z = float(q[2])
            feedback.orientation.w = float(q[3])
        feedback.distance_travelled     = float(self._mission_distance_m)
        feedback.coverage_area          = float(self._step_coverage_area)
        feedback.max_posterior          = float(self._step_max_posterior)
        feedback.ig_geometric           = float(self._step_ig_geometric)
        feedback.ig_semantic            = float(self._step_ig_semantic)
        feedback.vlm_inference_time_sec = float(self._step_vlm_time)
        feedback.target_estimate_valid  = self._step_target_est_x is not None
        feedback.target_estimate_x      = float(self._step_target_est_x or 0.0)
        feedback.target_estimate_y      = float(self._step_target_est_y or 0.0)
    
        goal_handle.publish_feedback(feedback)

    # --- STEP COMPONENTS ---

    def acquire_observation(self) -> bool:
        """
        Calls `acquire_observation` service (exposed by coverage_map_node) to get the current coverage polygon(s)
        and update the current geometric footprint that the camera sees (self.coverage).
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

        self._step_coverage_area = float(self.coverage.area) if self.coverage else 0.0
        # result.message contains both the total area and the number of connected components
        self.get_logger().info(f"[Geometric Map Updated]: {result.message}")
        return True
    
    def update_posterior(self) -> bool:
        """
        Calls the `update_posterior` service that asks the semantic_belief_tracker to update the 
        posterior field F based on the new observation and the current coverage (ROI).
        """
        if not self.update_posterior_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("/update_posterior service unavailable — skipping")
            return False

        req = UpdatePosterior.Request()

        # ROI: bbox of current coverage + margin, so the tracker only needs to publish the relevant slice of the field F
        if self.coverage is not None and not self.coverage.is_empty:
            xmin, ymin, xmax, ymax = self.coverage.bounds
            margin = self.active_domain_radius
            req.roi_x_min = xmin - margin
            req.roi_x_max = xmax + margin
            req.roi_y_min = ymin - margin
            req.roi_y_max = ymax + margin
        # Else, coverage is None and leave fields to 0.0, i.e., tracker publishes the whole field.

        result = self.update_posterior_client.call(req)
        if result is None or not result.success:
            msg = result.message if result else "no response"
            self.get_logger().warn(f"Posterior update failed: {msg}")
            return False

        # For action feedback
        self._step_max_posterior = float(result.max_posterior)
        if result.message == "TARGET_FOUND":
            self._step_target_est_x = float(result.target_estimate_x)
            self._step_target_est_y = float(result.target_estimate_y)
        else:
            self._step_target_est_x = None
            self._step_target_est_y = None

        self._step_vlm_time      = float(result.vlm_inference_time_sec)
        self.get_logger().info(f"Posterior update: {result.message}; VLM time: {self._step_vlm_time:.2f}s")
        return True

    def compute_active_domain(self):        
        if self.coverage is None or self.coverage.is_empty:
            if self.current_pose is not None:
                cx, cy = self.current_pose[0, 3], self.current_pose[1, 3]
            else:
                cx, cy = 0.0, 0.0
            return active_domain_fn(self.coverage, self.map_polygon, self.active_domain_radius, bootstrap_centre=(cx, cy), bootstrap_radius=self.active_domain_radius)
        
        return active_domain_fn(self.coverage, self.map_polygon, self.active_domain_radius)

    def generate_candidates(self, domain) -> list[tuple[float, float, float]]:
        """
        Generate candidates on the boundary of the cumulative coverage,
        plus one semantic candidate per detection (at μ_d, yaw toward μ_d).
        """
        # how much the camera footprint is offset forward from the drone's position? We want to sample candidates that 
        # are forward-facing relative to the current view, so we push them forward by this offset along the outward normal direction.
        overlap = self.frontier_spacing * 0.5
        offset = max(0.0, self._camera_forward_offset() - overlap)
        self.get_logger().info(f"[candidates] forward_offset={offset:.2f}m (raw={self._camera_forward_offset():.2f}m, overlap={overlap:.2f}m)")

        # Bootstrap for the first step
        if self.coverage is None or self.coverage.is_empty:
            current_fp = compute_ground_footprint(self.current_pose, self.camera, self.h_ground, self.max_projection_dist)
            if not current_fp.is_valid or current_fp.area < 0.1:
                return []
            return self._sample_boundary_with_outward_yaw(
                current_fp,
                self.frontier_spacing,
                outward_probe_dist=self.frontier_spacing,
                exclude_inside=None,
                forward_offset=offset
            )

        # Normal regime
        candidates = []
        polys = (self.coverage.geoms
                if isinstance(self.coverage, MultiPolygon)
                else [self.coverage])
        for poly in polys:
            candidates.extend(
                self._sample_boundary_with_outward_yaw(
                    poly,
                    self.frontier_spacing,
                    outward_probe_dist=self.frontier_spacing,
                    exclude_inside=None,
                    forward_offset=offset
                )
            )

        return candidates

    def _sample_boundary_with_outward_yaw(self, poly, spacing, outward_probe_dist, exclude_inside=None, forward_offset=0.0):
        """
        Sample equispaced points along the exterior boundary of `poly`, compute outward normals, and 
        skip points whose outward probe lands inside `exclude_inside` (used to discard candidates that would look
        back into already-covered area).

        Returns a list of (x, y, yaw) tuples with yaw = outward normal angle.
        """
        out = []
        bnd = poly.exterior # this is the perimeter along which we sample candidates
        L = bnd.length      # how long is the perimeter?
        n = max(4, int(L / spacing)) # how many candidates can we fit along the perimeter at the given spacing?
        for i in range(n):
            s = i / n * L
            pt = bnd.interpolate(s) # walk on the perimeter and place a candidate pt at regular intervals   
            pt_eps = bnd.interpolate((s + 0.5) % L)
            tx = pt_eps.x - pt.x
            ty = pt_eps.y - pt.y
            tnorm = np.hypot(tx, ty) + 1e-9
            nx, ny = ty / tnorm, -tx / tnorm
            test_pt = ShapelyPoint(pt.x + 0.5 * nx, pt.y + 0.5 * ny) # a test probe to determine if the normal points inward or outward
            if poly.contains(test_pt):
                nx, ny = -nx, -ny

            if exclude_inside is not None:
                probe = ShapelyPoint(
                    pt.x + outward_probe_dist * nx,
                    pt.y + outward_probe_dist * ny,
                )
                if exclude_inside.contains(probe):
                    continue

            yaw = float(np.arctan2(ny, nx))
            wx = pt.x - forward_offset * nx
            wy = pt.y - forward_offset * ny
            out.append((float(wx), float(wy), yaw))
        return out

    def evaluate_candidate(self, w_xyw, domain):
        x, y, yaw = w_xyw
        pose_w = self._candidate_pose(x, y, yaw)
        fp = compute_ground_footprint(pose_w, self.camera, self.h_ground, self.max_projection_dist)
        if not fp.is_valid:
            return 0.0, 0.0, 0.0

        roi = fp.intersection(domain)
        if roi.is_empty or roi.area <= 0.0:
            return 0.0, 0.0, 0.0

        # Geometric Information Gain: area of the new ROI that would be covered by this candidate pose
        ig_geometric = self.pi_0 * roi.area

        with self._field_lock:
            F   = None if self._field_F is None else self._field_F.copy()
            ox  = self._field_origin_x
            oy  = self._field_origin_y
            res = self._field_resolution
            nx  = self._field_width
            ny  = self._field_height

        # Semantic Information Gain: sum of the posterior probabilities in the ROI, weighted by cell value
        ig_semantic = 0.0
        if F is not None and res > 0.0:
            xmin, ymin, xmax, ymax = roi.bounds
            j0 = max(0, int((xmin - ox) / res))
            j1 = min(nx - 1, int((xmax - ox) / res))
            i0 = max(0, int((ymin - oy) / res))
            i1 = min(ny - 1, int((ymax - oy) / res))

            if j1 >= j0 and i1 >= i0:
                # Build coordinate grids for the bounding box slice
                js = np.arange(j0, j1 + 1)
                is_ = np.arange(i0, i1 + 1)
                cx_grid = ox + (js + 0.5) * res          # shape (W,)
                cy_grid = oy + (is_ + 0.5) * res         # shape (H,)
                CX, CY = np.meshgrid(cx_grid, cy_grid)   # shape (H, W)

                # Vectorized point-in-polygon using shapely's prepared geometry
                prepare(roi)
                mask = contains_xy(roi, CX.ravel(), CY.ravel()).reshape(CX.shape)

                cell_area   = res * res
                ig_semantic = float(F[i0:i1+1, j0:j1+1][mask].sum()) * cell_area

        total = ig_geometric + ig_semantic

        self.get_logger().info(f"Candidate: Geo={ig_geometric:.2f}, Sem={ig_semantic:.2f} => total={total:.2f}")
        return total, ig_geometric, ig_semantic

    def _candidate_pose(self, x, y, yaw_world) -> np.ndarray:
        if self.T_body_to_opt is None:
            return self.current_pose
        return build_candidate_pose(x, y, yaw_world, self.current_pose[2, 3], self.T_body_to_opt)

    def choose_next_waypoint(self):
        """Returns (best_xy, best_ig, best_yaw) or (None, 0, 0)."""

        # Compute the active domain (corona around the current coverage) to constrain candidate generation and evaluation.
        domain = self.compute_active_domain()
        if domain.is_empty:
            return None, 0.0, 0.0

        # Publish the active domain
        self.publish_domain(domain)

        # Generate candidates on the boundary of the cumulative coverage
        candidates = self.generate_candidates(domain)

        # Discard candidates that are outside the active domain (may happen when reacing map boundaries)
        #candidates = [c for c in candidates if domain.contains(ShapelyPoint(c[0], c[1]))]
        
        if not candidates:
            return None, 0.0, 0.0
        
        self.publish_candidates(candidates) # RViz

        # Evaluate candidates and pick the best according to J(w) = IG(w) - gamma * dist(current_pose, w)
        cx, cy = self.current_pose[0, 3], self.current_pose[1, 3]

        # We store all candidate evaluation results in a list
        all_data = []
        for w in candidates:
            x, y, yaw = w
            ig, ig_geo, ig_sem = self.evaluate_candidate(w, domain)
            dist = np.hypot(x - cx, y - cy)
            all_data.append(((x, y), ig, ig_geo, ig_sem, yaw, dist))

        best_J = -np.inf
        best = None

        # Here we compute the argmax iteratively
        for (w, ig, ig_geo, ig_sem, yaw, dist) in all_data: # NOTE: here w is a tuple (x, y) belonging to the candidate list 
            J = ig - self.gamma * dist
            if J > best_J:
                best_J = J
                best   = (w, ig, ig_geo, ig_sem, yaw)
            self.get_logger().info(f"Candidate at ({w[0]:.1f}, {w[1]:.1f}), yaw={np.degrees(yaw):.1f}° | IG={ig:.2f} (geo={ig_geo:.2f}, sem={ig_sem:.2f}) | dist={dist:.1f} => J={J:.2f}")

        self.publish_best_candidate((best[0], best[1], best[4]) if best is not None else None) # RViz

        # For action feedback
        if best is not None:
            self._step_ig_geometric = best[2]
            self._step_ig_semantic  = best[3]

        # *** DEBUG ***
        if self._debug and hasattr(self, '_debug_stats') and self._debug_stats:
            stats = np.array(self._debug_stats)
            roi_areas, ig_geoms, ig_sems, ig_exps, fp_areas = stats.T
            n_zero_roi = int((roi_areas <= 0.01).sum())
            n_zero_sem = int((ig_sems <= 0.01).sum())
            coverage_ratio = (roi_areas / np.maximum(fp_areas, 1e-6)).mean()
            self.get_logger().info(
                f"[DBG eval] roi/fp ratio avg={coverage_ratio:.2f} \n"
                f"zero-roi: {n_zero_roi}/{len(stats)} \n"
                f"zero-sem: {n_zero_sem}/{len(stats)} \n"
                f"ig_sem range=[{ig_sems.min():.2f},{ig_sems.max():.2f}] \n"
                f"ig_geom range=[{ig_geoms.min():.2f},{ig_geoms.max():.2f}] \n"
            )
            self._debug_stats = []

        if self._debug and best is not None:
            w, ig, ig_geo, ig_sem, yaw = best
            dist = np.hypot(w[0] - cx, w[1] - cy)
            self.get_logger().info(
                f"[w*] pos=({w[0]:.1f},{w[1]:.1f}), yaw={np.degrees(yaw):.1f}° | "
                f"dist={dist:.1f}m, penalty={self.gamma*dist:.2f} | "
                f"Geo={ig_geo:.2f}, Sem={ig_sem:.2f}, IG={ig:.2f} | J={best_J:.2f}"
            )
        # ******

        if best is None:
            return None, 0.0, 0.0
        return best[0], best[1], best[4] # best_xy, best_ig, best_yaw

    # -- MOVE COMMANDS ---

    def _within_bounds(self, x: float, y: float) -> bool:
        return (self.bounds_x_min <= x <= self.bounds_x_max and self.bounds_y_min <= y <= self.bounds_y_max)

    def arm_and_takeoff(self) -> bool:
        if self.get_parameter("toy_run").value:
            self.get_logger().info("[TOY RUN] arm_and_takeoff skipped")
            return True
        if not self.takeoff_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("TakeOff action server unavailable")
            return False

        goal = Takeoff.Goal()
        goal.takeoff_height = self.flight_altitude
        goal.takeoff_speed  = 1.0

        future = self.takeoff_client.send_goal_async(goal)
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Takeoff goal rejected")
            return False

        result_future = goal_handle.get_result_async()

        if not result_future.result().result.takeoff_success:
            self.get_logger().error("Takeoff failed")
            return False

        self.get_logger().info(f"Takeoff completed at {self.flight_altitude}m")
        return True

    def send_move_command(self, x, y, yaw, z=None):
        if self.get_parameter("toy_run").value:
            self.get_logger().info(f"[TOY RUN] move --> ({x:.2f}, {y:.2f}) yaw={np.degrees(yaw):.1f}°")
            return True
        if not self._within_bounds(x, y):
            self.get_logger().warn(f"Waypoint ({x:.2f}, {y:.2f}) out of bounds — skipping.")
            return False

        if not self.gotowp_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("GoToWaypoint action server unavailable")
            return False

        goal = GoToWaypoint.Goal()
        goal.target_pose.header.frame_id    = "world"
        goal.target_pose.header.stamp       = self.get_clock().now().to_msg()
        goal.target_pose.point.x            = float(x)
        goal.target_pose.point.y            = float(y)
        goal.target_pose.point.z            = float(z if z is not None else self.flight_altitude)
        goal.max_speed                      = 3.0
        goal.yaw_mode.mode                  = YawMode.FIXED_YAW
        goal.yaw_mode.angle                 = float(np.degrees(yaw))

        future = self.gotowp_client.send_goal_async(goal)
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error("GoToWaypoint goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        return True
    
    def send_rotate_command(self, yaw_rad: float) -> bool:
        if self.get_parameter("toy_run").value:
            self.get_logger().info(f"[TOY RUN] rotate --> {np.degrees(yaw_rad):.1f}°")
            return True
        if self.current_pose is None:
            return False
        cx = float(self.current_pose[0, 3])
        cy = float(self.current_pose[1, 3])
        cz = float(self.current_pose[2, 3])
        return self.send_move_command(cx, cy, yaw_rad, z=cz)
    
    def _verify_target(self) -> bool:
        # Save pre-descent pose to return to in case of false positive or verification failure
        pre_x   = float(self.current_pose[0, 3])
        pre_y   = float(self.current_pose[1, 3])
        pre_yaw = self.current_yaw

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

        dx = pre_x - target_x
        dy = pre_y - target_y
        dist = float(np.hypot(dx, dy))

        VERIFY_DISTANCE = 0.6 #[m]

        if dist < 0.1:
            # Limit case: if the drone is already exactly above the target, 
            # we retreat using the angle in which it was already looking
            ux = -np.cos(pre_yaw)
            uy = -np.sin(pre_yaw)
        else:
            # Direction vector normalized (length 1) from target to drone
            ux = dx / dist
            uy = dy / dist
        
        verify_x = target_x + ux * VERIFY_DISTANCE
        verify_y = target_y + uy * VERIFY_DISTANCE
        verify_yaw = float(np.arctan2(-uy, -ux))

        self.get_logger().info(f"[verify] Moving to ({verify_x:.1f}, {verify_y:.1f}) with yaw={np.degrees(verify_yaw):.1f}°")

        self._publish_verify_waypoint(verify_x, verify_y, verify_yaw) # Rviz

        # Descend above the target
        ok = self.send_move_command(verify_x, verify_y, verify_yaw, z=25)
        if not ok:
            self.get_logger().error("[verify] Failed to reach verification position")
            return False

        self._wait_until_at_waypoint(verify_x, verify_y)

        # Grab the image
        img = self._grab_image()
        if img is None:
            self.get_logger().error("[verify] Failed to grab image for verification")
            self._return_to_pre_descent(pre_x, pre_y, pre_yaw)
            return False

        # Call /verify_target
        if not self.verify_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("[verify] /verify_target unavailable")
            self._return_to_pre_descent(pre_x, pre_y, pre_yaw)
            return False

        req = VerifyTarget.Request()
        req.image        = img
        req.target_query = self.instruction
        result = self.verify_client.call(req)

        if result is None:
            self.get_logger().error("[verify] /verify_target call timed out")
            self._return_to_pre_descent(pre_x, pre_y, pre_yaw)
            return False

        self.get_logger().info(f"[verify] Result: {result.message}")

        if result.confirmed:
            self.get_logger().info("***🎯 TARGET VERIFIED at close range! ***")
            return True

        # False Positive: go back to the pre-descent position and continue exploring
        self.get_logger().info("[verify] False positive — returning to pre-descent position")
        self._publish_verify_waypoint(pre_x, pre_y, pre_yaw) # Rviz
        self._return_to_pre_descent(pre_x, pre_y, pre_yaw)
        return False

    def _return_to_pre_descent(self, x: float, y: float, yaw: float):
        """Returns to the pre-descent position."""
        self.get_logger().info(f"Returning to ({x:.1f},{y:.1f}) at {self.flight_altitude}m")
        self.send_move_command(x, y, yaw)

    # --- VISUALIZATION ---

    def publish_domain(self, domain):
        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "active_domain"
        m.id = 0
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale.x = 0.05
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
            m_sphere.header.frame_id = "world"
            m_sphere.ns = "pts"
            m_sphere.id = i
            m_sphere.type = Marker.SPHERE
            m_sphere.pose.position.x = float(x)
            m_sphere.pose.position.y = float(y)
            m_sphere.pose.position.z = self.h_ground + 0.1
            m_sphere.scale.x = m_sphere.scale.y = m_sphere.scale.z = 0.1
            m_sphere.color.r, m_sphere.color.g, m_sphere.color.b, m_sphere.color.a = 0.8, 0.8, 0.8, 1.0
            ma.markers.append(m_sphere)

            # --- Arrow (Yaw) ---
            m_arrow = Marker()
            m_arrow.header.frame_id = "world"
            m_arrow.ns = "yaw_arrows"
            m_arrow.id = i + 1000  # ID univoco
            m_arrow.type = Marker.ARROW
            
            # Positioning and orientation (via quaternions)
            m_arrow.pose.position.x = float(x)
            m_arrow.pose.position.y = float(y)
            m_arrow.pose.position.z = self.h_ground + 0.1
            
            # Converts yaw to quaternion (Z-axis rotation)
            m_arrow.pose.orientation.z = np.sin(yaw / 2.0)
            m_arrow.pose.orientation.w = np.cos(yaw / 2.0)
            
            # Size: x=length, y=width, z=height arrow
            m_arrow.scale.x, m_arrow.scale.y, m_arrow.scale.z = 0.2, 0.05, 0.05
            m_arrow.color.r, m_arrow.color.g, m_arrow.color.b, m_arrow.color.a = 1.0, 1.0, 1.0, 1.0 # White for the normal
            ma.markers.append(m_arrow)

        self.candidates_pub.publish(ma)
    
    def publish_best_candidate(self, best):
        mc_clear = Marker()
        mc_clear.header.frame_id = "world"
        mc_clear.header.stamp = self.get_clock().now().to_msg()
        mc_clear.ns = "chosen"
        mc_clear.action = Marker.DELETEALL
        self.chosen_pub.publish(mc_clear)

        if best is not None:
            w, ig, yaw = best
            mc = Marker()
            mc.header.frame_id = "world"
            mc.header.stamp = self.get_clock().now().to_msg()
            mc.ns = "chosen"
            mc.id = 0
            mc.type = Marker.ARROW
            mc.action = Marker.ADD
            
            arrow_length = 0.4
            start = Point(x=float(w[0]), y=float(w[1]), z=self.h_ground + 0.1)
            end = Point(
                x=float(w[0] + arrow_length * np.cos(yaw)),
                y=float(w[1] + arrow_length * np.sin(yaw)),
                z=self.h_ground + 0.1
            )
            
            mc.points = [start, end]
            mc.scale.x, mc.scale.y, mc.scale.z = 0.05, 0.10, 0.15
            mc.color.r, mc.color.g, mc.color.b, mc.color.a = 1.0, 1.0, 0.0, 1.0
            self.chosen_pub.publish(mc)

    def _publish_verify_waypoint(self, x: float, y: float, yaw: float):
        mc_clear = Marker()
        mc_clear.header.frame_id = "world"
        mc_clear.header.stamp    = self.get_clock().now().to_msg()
        mc_clear.ns              = "chosen"
        mc_clear.action          = Marker.DELETEALL
        self.chosen_pub.publish(mc_clear)

        mc = Marker()
        mc.header.frame_id = "world"
        mc.header.stamp    = self.get_clock().now().to_msg()
        mc.ns              = "chosen"
        mc.id              = 0
        mc.type            = Marker.ARROW
        mc.action          = Marker.ADD

        arrow_length = 0.4
        start = Point(x=float(x), y=float(y), z=self.h_ground + 0.1)
        end   = Point(
            x=float(x + arrow_length * np.cos(yaw)),
            y=float(y + arrow_length * np.sin(yaw)),
            z=self.h_ground + 0.1
        )
        mc.points = [start, end]
        mc.scale.x, mc.scale.y, mc.scale.z = 0.05, 0.10, 0.15
        mc.color.r, mc.color.g, mc.color.b, mc.color.a = 0.0, 1.0, 1.0, 1.0  # cyan
        self.chosen_pub.publish(mc)

    def _save_annotated_image(self, step: int):
        with self._annotated_image_lock:
            msg = self._annotated_image

        if msg is None:
            self.get_logger().warn(f"[save] No annotated image available at step {step}")
            return

        try:
            # Crea la cartella con timestamp al primo step
            if self._save_dir is None:
                base_dir = self.get_parameter("save_dir").value
                ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                self._save_dir = os.path.join(base_dir, ts)
                os.makedirs(self._save_dir, exist_ok=True)
                self.get_logger().info(f"[save] Saving images to: {self._save_dir}")

            bridge = CvBridge()
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            path = os.path.join(self._save_dir, f"step{step}.png")
            cv2.imwrite(path, cv_img)
            self.get_logger().info(f"[save] Annotated image saved to {path}")
        except Exception as e:
            self.get_logger().error(f"[save] Failed to save image at step {step}: {e}")

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