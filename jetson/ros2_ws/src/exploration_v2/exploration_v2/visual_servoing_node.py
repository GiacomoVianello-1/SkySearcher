import re
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
    
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger
from interfaces.srv import VerifyTarget, VisualServoingVerify
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from tf2_ros import Buffer, TransformListener
from .utils import lookup_transform_at, lookup_static_transform
from .pilot import ManualPilotMonitor, NavigationResult


def _stamp_to_sec(stamp) -> float:
    """builtin_interfaces/Time to float seconds."""
    return stamp.sec + stamp.nanosec * 1e-9


class VisualServoingNode(Node):
    def __init__(self):
        super().__init__("visual_servoing_node")

        self.declare_parameter("camera_frame",      "Drone1/bottom_center_optical")
        self.declare_parameter("world_frame",       "world")
        self.declare_parameter("odom_frame",        "Drone1")
        self.declare_parameter("rgb_topic",         "/airsim_node/Drone1/bottom_center_Scene/image")
        self.declare_parameter("depth_topic",       "/airsim_node/Drone1/bottom_center_DepthPlanar/image")
        self.declare_parameter("min_altitude",      5.0)
        self.declare_parameter("marker_scale", 1.0)
        self.declare_parameter("max_altitude",      35.0)    
        self.declare_parameter("standoff_distance", 4.0)
        self.declare_parameter("min_valid_depth",   0.5)
        self.declare_parameter("max_valid_depth",   10.0)
        self.declare_parameter("x_dist",            10.0)
        self.declare_parameter("vlm_frame_delay_tolerance", 1.0)  # Refuse to run the VLM on a frame older than this (s)
        self.declare_parameter("tf_timeout_sec",            0.1)  # How long to wait for the camera pose at the frame stamp

        # Manual pilot: how close the pilot must get before the loop continues
        self.declare_parameter("arrival_position_tolerance", 1.00)  # m, XY
        self.declare_parameter("arrival_altitude_tolerance", 0.50)  # m, Z
        self.declare_parameter("arrival_yaw_tolerance_deg",  20.0)  # deg, 180 disables the yaw check
        self.declare_parameter("arrival_hold_time_sec",       1.0)  # must stay in tolerance this long
        self.declare_parameter("viewpoint_timeout_sec",       0.0)  # 0 = hold until reached
        self.declare_parameter("rgb_depth_sync_tolerance",    0.1)  # s, max RGB/depth stamp gap


        self.camera_frame      = str(self.get_parameter("camera_frame").value)
        self.world_frame       = str(self.get_parameter("world_frame").value)
        self.odom_frame        = str(self.get_parameter("odom_frame").value)
        self.rgb_topic         = str(self.get_parameter("rgb_topic").value)
        self.depth_topic       = str(self.get_parameter("depth_topic").value)
        self.min_altitude      = float(self.get_parameter("min_altitude").value)
        self.max_altitude      = float(self.get_parameter("max_altitude").value)        
        self.standoff_distance = float(self.get_parameter("standoff_distance").value)
        self.min_valid_depth   = float(self.get_parameter("min_valid_depth").value)
        self.max_valid_depth   = float(self.get_parameter("max_valid_depth").value)
        self.x_dist            = float(self.get_parameter("x_dist").value)
        self.marker_scale = float(self.get_parameter("marker_scale").value)
        self.vlm_frame_delay_tolerance = float(self.get_parameter("vlm_frame_delay_tolerance").value)
        self.tf_timeout_sec            = float(self.get_parameter("tf_timeout_sec").value)

        self.arrival_pos_tol      = float(self.get_parameter("arrival_position_tolerance").value)
        self.arrival_alt_tol      = float(self.get_parameter("arrival_altitude_tolerance").value)
        self.arrival_yaw_tol_deg  = float(self.get_parameter("arrival_yaw_tolerance_deg").value)
        self.arrival_hold_sec     = float(self.get_parameter("arrival_hold_time_sec").value)
        self.viewpoint_timeout_sec = float(self.get_parameter("viewpoint_timeout_sec").value)
        self.rgb_depth_sync_tolerance = float(self.get_parameter("rgb_depth_sync_tolerance").value)

        # (image, header.stamp) pairs, replaced as a single tuple so the two cannot come from different frames.
        self.latest_rgb   = None
        self.latest_depth = None
        self._last_move_ok = True
        self._abort_requested = False
        self.bridge       = CvBridge()

        self.cb_group = ReentrantCallbackGroup()

        # TF Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        self.rgb_sub = self.create_subscription(Image, self.rgb_topic, self.rgb_callback, 10, callback_group=self.cb_group)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10, callback_group=self.cb_group)

        # The drone is flown by a human: each servoing step waits for arrival.
        self.pilot = ManualPilotMonitor(
            self, self.tf_buffer, self.world_frame, self.odom_frame,
            position_tolerance=self.arrival_pos_tol,
            altitude_tolerance=self.arrival_alt_tol,
            yaw_tolerance_deg=self.arrival_yaw_tol_deg,
            hold_time_sec=self.arrival_hold_sec,
            abort_fn=lambda: self._abort_requested,
        )

        # Clients
        self.verify_vlm_client = self.create_client(VerifyTarget, "/verify_target", callback_group=self.cb_group)

        # Services
        self.servoing_srv = self.create_service(VisualServoingVerify, "/visual_servoing_verify", self.handle_visual_servoing_verify, callback_group=self.cb_group)
        # The planner calls verification synchronously, so /stop_mission cannot reach
        # a hold in progress. This is the only way out of one.
        self.abort_srv = self.create_service(Trigger, "/abort_servoing", self.handle_abort, callback_group=self.cb_group)

        # Publishers
        self.annotated_pub = self.create_publisher(Image,       "/annotated_img", 10)
        self.marker_pub = self.create_publisher(MarkerArray,    "/triangulation_markers", 10)
        self.annotated_depth_pub = self.create_publisher(Image, "/annotated_depth", 10)

        self.get_logger().info("Visual Servoing Node ready with Depth-Guided Closed-Loop.")
    
    def handle_abort(self, request, response):
        self._abort_requested = True
        self.get_logger().warn("Abort requested — giving up on the current viewpoint.")
        response.success = True
        response.message = "Servoing abort requested"
        return response

    def rgb_callback(self, msg: Image):
        self.latest_rgb = (self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"), msg.header.stamp)

    def depth_callback(self, msg: Image):
        self.latest_depth = (self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1"), msg.header.stamp)

    def _frame_age_sec(self, stamp) -> float:
        return (self.get_clock().now() - rclpy.time.Time.from_msg(stamp)).nanoseconds * 1e-9

    def _grab_images(self) -> tuple[np.ndarray | None, np.ndarray | None, object | None]:
        """Returns (rgb, depth, rgb_stamp), or (None, None, None) if either frame is missing or stale."""
        rgb, depth = self.latest_rgb, self.latest_depth
        if rgb is None or depth is None:
            self.get_logger().warn("Images are not available on RGB/Depth topics.")
            return None, None, None

        for name, (_, stamp) in (("RGB", rgb), ("Depth", depth)):
            age = self._frame_age_sec(stamp)
            if age > self.vlm_frame_delay_tolerance:
                self.get_logger().warn(f"{name} frame is {age:.2f}s old (> {self.vlm_frame_delay_tolerance:.2f}s) — discarding.")
                return None, None, None

        skew = abs(_stamp_to_sec(rgb[1]) - _stamp_to_sec(depth[1]))
        if skew > self.rgb_depth_sync_tolerance:
            self.get_logger().warn(f"RGB/depth are {skew:.2f}s apart (> {self.rgb_depth_sync_tolerance:.2f}s) — discarding.")
            return None, None, None

        return rgb[0], depth[0], rgb[1]

    def send_move_command(self, x: float, y: float, z: float, yaw_rad: float, label: str = "servoing viewpoint") -> bool:
        """Ask the pilot for a viewpoint and block until they hold it."""
        z = max(self.min_altitude, min(self.max_altitude, z))
        res = self.pilot.wait_until_at(x, y, z, yaw_rad,
                                       timeout_sec=self.viewpoint_timeout_sec, label=label)
        return res == NavigationResult.SUCCESS

    def query_vlm_verification(self, rgb_img: np.ndarray, target_query: str, object_desc: str) -> tuple[list[int], bool]:
        """Queries VLM to verify if target is present and extract 2D BBox."""
        if not self.verify_vlm_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/verify_target unavailable")
            return [], False

        req = VerifyTarget.Request()
        req.image = self.bridge.cv2_to_imgmsg(rgb_img, encoding="bgr8")
        req.target_query = target_query
        req.object_description = object_desc

        res = self.verify_vlm_client.call(req)
        if res is None:
            return [], False

        msg_upper = res.message.upper()
        bbox_2d = []

        match_bbox = re.search(r'BBOX:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', msg_upper)
        if match_bbox:
            bbox_2d = [int(match_bbox.group(1)), int(match_bbox.group(2)), int(match_bbox.group(3)), int(match_bbox.group(4))]

        return bbox_2d, res.confirmed

    def _extract_depth_from_bbox(self, depth_img_in_meters: np.ndarray, bbox_2d: list[int]) -> float | None:
        h_depth, w_depth = depth_img_in_meters.shape

        ymin, xmin, ymax, xmax = bbox_2d
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0

        cx_d = int((cx / 1000.0) * w_depth)
        cy_d = int((cy / 1000.0) * h_depth)

        cx_d = max(0, min(w_depth - 1, cx_d))
        cy_d = max(0, min(h_depth - 1, cy_d))

        depth_val = float(depth_img_in_meters[cy_d, cx_d])

        # Discard implausible readings: NaN/inf or outside the physically sensible range for this approach scenario.
        # An invalid depth is treated as "measure not available" (None), so the caller falls back to the existing fallback
        # ("blind step") instead of trusting a junk number that would send the drone in an absurd direction.
        if not np.isfinite(depth_val) or not (self.min_valid_depth <= depth_val <= self.max_valid_depth):
            self.get_logger().warn(f"Out-of-range value ({depth_val}), expected in [{self.min_valid_depth}, {self.max_valid_depth}]m. Discarding measure.")
            self.publish_annotated_depth(depth_img_in_meters, cx_d, cy_d, None)
            return None
 
        self.publish_annotated_depth(depth_img_in_meters, cx_d, cy_d, depth_val)

        return depth_val

    def _get_drone_target_pose(self, p_cam_desired: np.ndarray, yaw_desired: float) -> tuple[float, float, float]:
        T_body_to_opt = lookup_static_transform(self.tf_buffer, self.odom_frame, self.camera_frame)
        if T_body_to_opt is not None:
            t_body_to_opt = T_body_to_opt[:3, 3]
            c_y, s_y = np.cos(yaw_desired), np.sin(yaw_desired)
            R_yaw = np.array([
                [c_y, -s_y, 0.0],
                [s_y,  c_y, 0.0],
                [0.0,  0.0, 1.0]
            ])
            p_body_desired = p_cam_desired - R_yaw @ t_body_to_opt
            return float(p_body_desired[0]), float(p_body_desired[1]), float(p_body_desired[2])
        else:
            return float(p_cam_desired[0]), float(p_cam_desired[1]), float(p_cam_desired[2])

    def _process_viewpoint(self, x: float, y: float, z: float, yaw: float, step_num: int, max_steps: int, target_query: str, object_desc: str) -> tuple[bool, list[int], float, np.ndarray | None, np.ndarray | None, float | None]:
        self._last_move_ok = self.send_move_command(x, y, z, yaw, label=f"servoing step {step_num}/{max_steps}")
        if not self._last_move_ok:
            self.get_logger().error(f"❌ Step {step_num}: gave up on ({x:.1f}, {y:.1f}, z={z:.1f}m)")
            return False, [], 0.0, None, None, None

        # Acquire rgbd images
        rgb_img, depth_img, rgb_stamp = self._grab_images()

        if rgb_img is None or depth_img is None:
            self.get_logger().error(f"❌ Step {step_num}: Image capture failed")
            return False, [], 0.0, None, None, None

        # Ask VLM for bbox and confirmation
        bbox_2d, confirmed = self.query_vlm_verification(rgb_img, target_query, object_desc)

        measured_depth = None
        if confirmed:
            if len(bbox_2d) == 4:
                measured_depth = self._extract_depth_from_bbox(depth_img, bbox_2d)
            else:
                # Fallback: target confirmed but no bbox. Use center of depth image.
                h_depth, w_depth = depth_img.shape
                cx_d = w_depth // 2
                cy_d = h_depth // 2
                depth_val = float(depth_img[cy_d, cx_d])
                if np.isfinite(depth_val) and (self.min_valid_depth <= depth_val <= self.max_valid_depth):
                    measured_depth = depth_val
                    self.get_logger().info(f"⚠️ Target confirmed but bbox empty/invalid. Using center depth: {measured_depth:.2f}m")
                else:
                    self.get_logger().warn(f"Center depth value ({depth_val}) out of range. Discarding.")

        self.publish_annotated_image(rgb_img, target_query, step_num, max_steps, confirmed, z, x, y, bbox_2d, measured_depth)

        o_world = None
        d_world = None
        c_x_offset = 0.0
        if confirmed:
            if len(bbox_2d) == 4:
                ymin, xmin, ymax, xmax = bbox_2d
                c_x_offset = ((xmin + xmax) / 2000.0) - 0.5

                h_img, w_img = rgb_img.shape[:2]
                fx = fy = w_img / 2.0
                cx_img, cy_img = w_img / 2.0, h_img / 2.0

                u_center = ((xmin + xmax) / 2000.0) * w_img
                v_center = ((ymin + ymax) / 2000.0) * h_img

                x_opt = (u_center - cx_img) / fx
                y_opt = (v_center - cy_img) / fy
                z_opt = 1.0
            else:
                # Fallback ray along camera's optical axis
                c_x_offset = 0.0
                x_opt = 0.0
                y_opt = 0.0
                z_opt = 1.0

            d_opt = np.array([x_opt, y_opt, z_opt])
            norm_opt = np.linalg.norm(d_opt)
            if norm_opt > 1e-6:
                d_opt = d_opt / norm_opt

            # Pose at the RGB stamp, not the latest: the bbox was detected in that frame.
            T_cam = lookup_transform_at(self.tf_buffer, self.world_frame, self.camera_frame, rgb_stamp, self.tf_timeout_sec)
            if T_cam is None:
                self.get_logger().warn(f"No {self.world_frame} -> {self.camera_frame} transform at the RGB stamp — no ray this step.")
            else:
                o_world = T_cam[:3, 3]
                d_world = T_cam[:3, :3] @ d_opt
                norm_w = np.linalg.norm(d_world)
                if norm_w > 1e-6:
                    d_world = d_world / norm_w

        self.get_logger().info(f"🔍 Step {step_num}: VLM Confirmed={confirmed}, Target depth={measured_depth}m")
        return confirmed, bbox_2d, c_x_offset, o_world, d_world, measured_depth

    def handle_visual_servoing_verify(self, request: VisualServoingVerify.Request, response: VisualServoingVerify.Response) -> VisualServoingVerify.Response:
        # The planner calls this service and fills out these parameters
        target_x = request.target_x
        target_y = request.target_y
        start_x = request.pre_x
        start_y = request.pre_y
        start_z = request.pre_z
        target_query = request.target_query
        object_desc = request.object_description
        max_steps = int(request.max_adjust_steps)
        self._abort_requested = False

        curr_x = start_x
        curr_y = start_y
        drone_z = start_z

        dx = target_x - start_x
        dy = target_y - start_y
        total_dist = float(np.hypot(dx, dy))
        if total_dist > 0.1:
            curr_yaw = float(np.arctan2(dy, dx))
        else:
            curr_yaw = request.pre_yaw

        step_id = 1
        last_confirmed = False # tracks if the last processed step has confirmed the target 

        while step_id <= max_steps:
            self.get_logger().info("**********************************")
            self.get_logger().info(f"***** Servoing Loop Step {step_id}/{max_steps} *****")
            self.get_logger().info("**********************************")
            
            confirmed, bbox_2d, c_x, o_world, d_world, measured_depth = self._process_viewpoint(curr_x, curr_y, drone_z, curr_yaw, step_id, max_steps, target_query, object_desc)
            last_confirmed = confirmed

            if not self._last_move_ok:
                self.get_logger().error("❌ Gave up on the servoing viewpoint. Aborting verification.")
                self.send_move_command(start_x, start_y, start_z, request.pre_yaw, label="pre-descent pose")
                response.verified = False
                response.final_x, response.final_y, response.final_z = start_x, start_y, start_z
                response.final_yaw = request.pre_yaw
                response.message = f"Pilot did not reach the viewpoint at step {step_id}"
                return response


            if confirmed and len(bbox_2d) == 4 and abs(c_x) > 0.03:
                angle_offset_rad = float(np.arctan(2.0 * c_x * np.tan(np.radians(45.0))))
                curr_yaw -= angle_offset_rad
                self.send_move_command(curr_x, curr_y, drone_z, curr_yaw, label="heading correction")

            if not confirmed or o_world is None or d_world is None:
                if step_id == 1:
                    self.get_logger().info("⚠️  Target not confirmed at step 1. Using candidate coordinates fallback ray.")
                    o_world = np.array([curr_x, curr_y, drone_z])
                    d_dir = np.array([target_x - curr_x, target_y - curr_y, 0.0 - drone_z])
                    d_norm = np.linalg.norm(d_dir)
                    if d_norm > 1e-6:
                        d_world = d_dir / d_norm
                    else:
                        d_world = np.array([0.0, 0.0, -1.0])
                else:
                    self.get_logger().info("❌  Target lost during approach loop. Aborting.")
                    self.send_move_command(start_x, start_y, start_z, request.pre_yaw, label="pre-descent pose")
                    response.verified = False
                    response.final_x = start_x
                    response.final_y = start_y
                    response.final_z = start_z
                    response.final_yaw = request.pre_yaw
                    response.message = f"Target lost at step {step_id} (was confirmed earlier, not a false positive)"
                    return response

            self.publish_ray_marker(o_world, d_world, step_id)

            if measured_depth is not None:
                dist_to_move = measured_depth - self.standoff_distance
                #self.get_logger().info(f" Target depth detected: {measured_depth:.2f}m. Dist to move: {dist_to_move:+.2f}m")
                
                if abs(dist_to_move) < 0.1:
                    self.get_logger().info(f"🎯 Standoff distance of {self.standoff_distance}m achieved! Starting final verification.")
                    final_confirmed, _, _, _, _, final_depth = self._process_viewpoint(
                        curr_x, curr_y, drone_z, curr_yaw, step_id + 1, max_steps, target_query, object_desc
                    )
                    if final_confirmed:
                        self.get_logger().info(f"🎯 TARGET VERIFIED AT {self.standoff_distance}m STANDOFF!")
                        response.verified = True
                        response.final_x = curr_x
                        response.final_y = curr_y
                        response.final_z = drone_z
                        response.final_yaw = curr_yaw
                        response.message = f"Verified with depth feedback at {final_depth:.2f}m"
                        return response
                    else:
                        self.get_logger().info("❌ Final verification failed at close range.")
                        self.send_move_command(start_x, start_y, start_z, request.pre_yaw, label="pre-descent pose")
                        response.verified = False
                        response.final_x = start_x
                        response.final_y = start_y
                        response.final_z = start_z
                        response.final_yaw = request.pre_yaw
                        response.message = "Failed close-range final verification"
                        return response

                dist_to_move_clamped = max(-10.0, min(10.0, dist_to_move))
                p_new = o_world + d_world * dist_to_move_clamped
            else:
                self.get_logger().info(f"⚠️  Target depth unavailable. Blind step along ray of {self.x_dist}m.")
                p_new = o_world + d_world * self.x_dist

            yaw_target = float(np.arctan2(target_y - p_new[1], target_x - p_new[0]))
            x_drone, y_drone, z_drone = self._get_drone_target_pose(p_new, yaw_target)
            z_drone = max(self.min_altitude, min(self.max_altitude, z_drone))
            
            curr_x = x_drone
            curr_y = y_drone
            drone_z = z_drone
            curr_yaw = yaw_target
            step_id += 1

        if last_confirmed:
            # Target has been confirmed by the VLM at the last processed step. The verification
            # criterion is the VLM confirmation, not the precision of the standoff distance: the
            # distance is only a refinement of the final position. Therefore, we remain in the
            # current position (where the target is confirmed) instead of returning to the start.
            self.get_logger().info(f"✅ Max steps reached but target confirmed at final step (standoff distance {self.standoff_distance}m not precisely reached). Accepting as verified.")
            response.verified = True
            response.final_x = curr_x
            response.final_y = curr_y
            response.final_z = drone_z
            response.final_yaw = curr_yaw
            response.message = "Verified: target confirmed at final step, standoff distance not precisely reached within max steps"
        else:
            # Last processed step: target NOT confirmed. With all steps exhausted and
            # no confirmation at the last attempt, it is considered a false positive candidate.
            self.get_logger().info("❌ Max steps reached: target not confirmed at final step. Likely false positive.")
            self.send_move_command(start_x, start_y, start_z, request.pre_yaw, label="pre-descent pose")
            response.verified = False
            response.final_x = start_x
            response.final_y = start_y
            response.final_z = start_z
            response.final_yaw = request.pre_yaw
            response.message = "FALSE_POSITIVE: target not confirmed at final step after exhausting all steps"
 
        return response

    # --- VISUALIZATION HELPERS ---
    def publish_ray_marker(self, o_world: np.ndarray, d_world: np.ndarray, step_id: int):
        """This is the ray joining the optical frame of the drone camera with the center of the bbox."""
        ma = MarkerArray()
        m_ray = Marker()
        m_ray.header.frame_id = self.world_frame
        m_ray.header.stamp = self.get_clock().now().to_msg()
        m_ray.ns = "visual_servoing_rays"
        m_ray.id = step_id
        m_ray.type = Marker.LINE_LIST
        m_ray.action = Marker.ADD
        m_ray.scale.x = 0.35 * self.marker_scale
        m_ray.color.r = 1.0
        m_ray.color.g = 0.647
        m_ray.color.b = 0.0
        m_ray.color.a = 1.0
        p_start = Point(x=float(o_world[0]), y=float(o_world[1]), z=float(o_world[2]))
        p_end_vec = o_world + d_world * 70.0 # This is an hardcoded value for the ray length.
        p_end = Point(x=float(p_end_vec[0]), y=float(p_end_vec[1]), z=float(p_end_vec[2]))
        m_ray.points.append(p_start)
        m_ray.points.append(p_end)
        ma.markers.append(m_ray)
        self.marker_pub.publish(ma)

    def publish_annotated_image(self, cv_img: np.ndarray, target_query: str, step: int, max_steps: int, confirmed: bool, drone_z: float, curr_x: float, curr_y: float, bbox_2d: list[int], depth_val: float | None):
        """Draws bounding box and HUD telemetry on image, then publishes to /annotated_img."""
        annotated_img = cv_img.copy()
        h, w = annotated_img.shape[:2]

        if len(bbox_2d) == 4 and any(c > 0 for c in bbox_2d):
            ymin, xmin, ymax, xmax = bbox_2d
            x1 = int((xmin / 1000.0) * w)
            y1 = int((ymin / 1000.0) * h)
            x2 = int((xmax / 1000.0) * w)
            y2 = int((ymax / 1000.0) * h)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            depth_str = f" | Depth: {depth_val:.2f}m" if depth_val is not None else " | Depth: Out of range"
            cv2.putText(annotated_img, f"{target_query}{depth_str}", (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cx, cy = w // 2, h // 2
            box_sz = 90
            cv2.rectangle(annotated_img, (cx - box_sz, cy - box_sz), (cx + box_sz, cy + box_sz), (255, 200, 0), 2)

        hud_confirmed = "CONFIRMED" if confirmed else "UNCONFIRMED"
        banner_str = f"STEP {step}/{max_steps} | {hud_confirmed} | ALT: {drone_z:.1f}m | POS: ({curr_x:.1f},{curr_y:.1f})"
        cv2.rectangle(annotated_img, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(annotated_img, banner_str, (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        out_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding="bgr8")
        out_msg.header.frame_id = self.camera_frame
        out_msg.header.stamp = self.get_clock().now().to_msg()
        self.annotated_pub.publish(out_msg)

    def publish_annotated_depth(self, depth_data: np.ndarray, cx_d: int, cy_d: int, depth_val: float | None):
        """Draws the point onto the depth image and reports useful metrics."""
        vis_range_m = self.max_valid_depth
        h, w = depth_data.shape
        clipped = np.clip(depth_data, 0.0, vis_range_m)
        normalized = (clipped / vis_range_m * 255.0).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        scale = 3 # we incrase the resolution to better visualize it  
        colored = cv2.resize(colored, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

        px, py = cx_d * scale, cy_d * scale
        cv2.drawMarker(colored, (px, py), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.circle(colored, (px, py), 8, (0, 0, 255), 2)
        depth_str = f"{depth_val:.2f} m" if depth_val is not None else "INVALID"
        cv2.putText(colored, f"pixel=({cx_d},{cy_d})  depth={depth_str}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out_msg = self.bridge.cv2_to_imgmsg(colored, encoding="bgr8")
        out_msg.header.frame_id = self.camera_frame
        out_msg.header.stamp = self.get_clock().now().to_msg()
        self.annotated_depth_pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = VisualServoingNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()