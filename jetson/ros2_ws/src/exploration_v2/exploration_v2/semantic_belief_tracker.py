import threading
import numpy as np
import cv2
import json

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from cv_bridge import CvBridge

from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point, Pose2D
from std_msgs.msg import Bool, Header

from interfaces.msg import PosteriorField as PosteriorFieldMsg, CoverageField, DetectionPositions
from interfaces.srv import DetectSemantics, UpdatePosterior, EvaluateWaypoints, ResetBelief

from tf2_ros import Buffer, TransformListener

from .camera import Camera
from .utils import lookup_transform_at, compute_ground_footprint, project_bbox_montecarlo, lookup_static_transform, build_candidate_pose
from .active_domain import rasterize_polygon
from .grid_bayes_filter import GridBayesFilter
from shapely.geometry import Polygon
from shapely import prepare, contains_xy

from dataclasses import dataclass

@dataclass
class _BlobViz:
    """Minimal blob for RViz markers visualization."""
    x: float        # X coordinate of the blob center
    y: float        # Y coordinate of the blob center
    scale_x: float  # Length of the major axis (ellipse)
    scale_y: float  # Length of the minor axis (ellipse)
    yaw: float      # Rotation angle of the ellipse
    label: str      # Label of the blob (for debugging/visualization)


class SemanticMapNode(Node):
    def __init__(self):
        super().__init__("semantic_map_node")

        # --- PARAMETERS ---
        self.declare_parameter("camera_frame",      "Drone1/bottom_center_optical")
        self.declare_parameter("camera_info_topic", "/airsim_node/Drone1/bottom_center_Scene/camera_info")
        self.declare_parameter("rgb_topic",         "/airsim_node/Drone1/bottom_center_Scene/image")
        self.declare_parameter("odom_topic",        "/airsim_node/Drone1/odom_local")
        self.declare_parameter("odom_frame",        "Drone1")
        self.declare_parameter("world_frame",       "world")

        self.declare_parameter("instruction", "")

        self.declare_parameter("h_ground", 0.0) # Height of the ground plane for ray-casting (flat earth assumption)
        self.declare_parameter("max_projection_dist", 200.0)

        self.declare_parameter("vlm_frame_delay_tolerance", 1.0)  # Refuse to run the VLM on a frame older than this (s)
        self.declare_parameter("tf_timeout_sec", 0.1)             # How long to wait for the camera pose at the frame stamp

        # Grid map
        self.declare_parameter("grid_x_min", -350.0)
        self.declare_parameter("grid_x_max",  350.0)
        self.declare_parameter("grid_y_min", -350.0)
        self.declare_parameter("grid_y_max",  350.0)
        self.declare_parameter("delta_grid", 0.8)    # Resolution of the grid map (meters per cell)
        self.declare_parameter("nu", 5.0)

        self.declare_parameter("active_domain_radius", 24.0) # Thickness of corona ring D_t
        self.declare_parameter("marker_scale", 1.0)
        self.marker_scale = float(self.get_parameter("marker_scale").value)

        self.declare_parameter("gamma_context", 0.5)
        self.declare_parameter("alpha_obs", 0.1)    # Multiplicative discount alpha_obs for repeated observations.
        self.declare_parameter("beta_absent", 0.1)  # Target non-observation penalty factor for context footprints. Set to 0 to disable.

        # Resolve parameters
        self.camera_frame = self.get_parameter("camera_frame").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.world_frame = self.get_parameter("world_frame").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.instruction = str(self.get_parameter("instruction").value)

        self.h_ground = float(self.get_parameter("h_ground").value)
        self.max_projection_dist = float(self.get_parameter("max_projection_dist").value)
        self.vlm_frame_delay_tolerance = float(self.get_parameter("vlm_frame_delay_tolerance").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)

        x_min = float(self.get_parameter("grid_x_min").value)
        x_max = float(self.get_parameter("grid_x_max").value)
        y_min = float(self.get_parameter("grid_y_min").value)
        y_max = float(self.get_parameter("grid_y_max").value)
        delta_grid = float(self.get_parameter("delta_grid").value)

        gamma_context = float(self.get_parameter("gamma_context").value)
        nu = float(self.get_parameter("nu").value)
        alpha_obs = float(self.get_parameter("alpha_obs").value)
        beta_absent = float(self.get_parameter("beta_absent").value)

        self._bridge = CvBridge()
        self.cb_group = ReentrantCallbackGroup()

        # --- INITIALIZATION OF THE BAYES FIELD ---
        self._field = GridBayesFilter(
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, delta=delta_grid, 
            nu=nu, gamma_context=gamma_context, alpha_obs=alpha_obs, beta_absent=beta_absent
        )
        self.map_polygon = Polygon([
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
        ])
        self.get_logger().info(f"Bayes Filter Grid Initialized ({self._field.nx}x{self._field.ny}), Uniform Prior over {self._field.N} cells: {self._field.pi_0}")

        # --- INTERNAL STATE ---
        self.camera = None
        self._latest_frame = None
        self._latest_frame_stamp = None  # header.stamp of _latest_frame; the pose must be looked up at it
        self._latest_frame_lock = threading.Lock()
        self.current_position = None

        self._prompts_lock = threading.Lock()
        self._field_lock = threading.RLock()   # _field is swapped on reset while the viz timer reads it
        self._instruction = ""
        self._primary_object = ""
        self._label_to_sigma = {}  # label -> sigma(json) mapping
        self._target_query = ""    # Unified string for the VLM
        self._latest_coverage_mask: np.ndarray | None = None
        self._coverage_mask_lock = threading.Lock()

        self.T_body_to_opt = None # Static transform from body frame to optical frame

        # --- TF AND SUBSCRIBERS ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(Image, self.rgb_topic, self.image_callback, 5)
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(CoverageField, "/coverage_field", self._coverage_field_cb, 1)

        # --- CLIENT AND SERVICES ---
        self.vlm_client = self.create_client(DetectSemantics, 'detect_semantic_clues', callback_group=self.cb_group)
        self.update_srv = self.create_service(UpdatePosterior, "/update_posterior", self.update_posterior_callback, callback_group=self.cb_group)
        self.evaluate_srv = self.create_service(EvaluateWaypoints, "/evaluate_waypoints", self.evaluate_waypoints_callback, callback_group=self.cb_group)
        self.reset_srv = self.create_service(ResetBelief, "/reset_semantic_belief", self.reset_semantic_belief_callback, callback_group=self.cb_group)

        # --- PUBLISHERS ---
        self.annotated_pub              = self.create_publisher(Image, "/annotated_img", 5)
        self.field_pub                  = self.create_publisher(PosteriorFieldMsg, "/posterior_field", 1)
        self.blobs_pub                  = self.create_publisher(MarkerArray, "/semantic_blobs", 5)
        self.target_estimate_pub        = self.create_publisher(Marker, "/target_estimate_marker", 1)
        self.grid_pub                   = self.create_publisher(OccupancyGrid, "/posterior_grid", 1)
        self.projected_points_pub       = self.create_publisher(PointCloud2, "/projected_bbox_points", 5)
        self.target_detected_pub        = self.create_publisher(Bool, "/target_detected", 10)
        self.detection_positions_pub    = self.create_publisher(DetectionPositions, "/detection_positions", 1)

        self._load_mission_config()

        # Publish initial uniform prior state at startup & maintain periodic viz timer for RViz
        self.create_timer(1.0, self._periodic_viz_timer_callback)
        self._periodic_viz_timer_callback()

        # Load the static transform from body frame to optical frame
        self.tf_timer = self.create_timer(0.1, self._initialize_tf)

        self.get_logger().info("Semantic Map Node ready.")
        
    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------
    def _periodic_viz_timer_callback(self):
        """Periodically publishes current posterior grid and field to RViz."""
        self._publish_grid_for_rviz()
        self._publish_field()

    def _initialize_tf(self):
        T = lookup_static_transform(self.tf_buffer, self.odom_frame, self.camera_frame)
        if T is not None: 
            self.T_body_to_opt = T
            self.get_logger().info("Static transform body->optical loaded via TF.")
            self.tf_timer.cancel()
        else:
            self.get_logger().warn("Static transform not yet available, retrying...")

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera is None:
            self.camera = Camera(
                h=msg.height, 
                w=msg.width, 
                fx=msg.k[0], 
                fy=msg.k[4],
                cx=msg.k[2], 
                cy=msg.k[5], 
                depth=self.max_projection_dist
            )

    def image_callback(self, msg: Image):
        cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self._latest_frame_lock:
            self._latest_frame = cv_img
            self._latest_frame_stamp = msg.header.stamp

    def odom_callback(self, msg: Odometry):
        self.current_position = msg.pose.pose.position

    def _coverage_field_cb(self, msg: CoverageField):
        """
        Receives the authoritative coverage grid from coverage_map_node and
        caches it as a boolean numpy mask for use in update_evidence().
        The grid is already rasterised — no Shapely needed here.
        """
        grid = np.array(msg.grid, dtype=np.float32).reshape(msg.height, msg.width)
        mask = grid > 0.5
        with self._coverage_mask_lock:
            self._latest_coverage_mask = mask

    def _load_mission_config(self):
        """Initializes mission configuration defaults (updated dynamically via ResetBelief service)."""
        primary = self.instruction if self.instruction else "target"
        self._instruction = self.instruction
        self._primary_object = primary
        self._label_to_sigma = {primary: 0.0}
        self._label_to_d_star = {primary: 0.0}
        self._target_query = primary

    def reset_semantic_belief_callback(self, request, response):
        """Resets the Bayesian belief field and loads new episode VLM parameters."""
        with self._prompts_lock:
            primary = request.primary_object if request.primary_object else self.instruction
            full_inst = getattr(request, 'full_instruction', primary)
            surroundings = json.loads(request.surroundings_json) if request.surroundings_json else []
            sigmas = json.loads(request.sigmas_json) if request.sigmas_json else []
            d_stars = json.loads(request.d_stars_json) if request.d_stars_json else []

            self._primary_object = primary
            self._full_instruction = full_inst if full_inst else primary
            self._label_to_sigma = {primary: 0.0}
            self._label_to_d_star = {primary: 0.0}

            for s, sig, d_star in zip(surroundings, sigmas, d_stars):
                label_pulita = s.strip()
                self._label_to_sigma[label_pulita] = float(sig)
                self._label_to_d_star[label_pulita] = float(d_star)

            self._target_query = ", ".join([primary] + surroundings)

            x_min = request.grid_x_min if (request.grid_x_max > request.grid_x_min) else float(self.get_parameter("grid_x_min").value)
            x_max = request.grid_x_max if (request.grid_x_max > request.grid_x_min) else float(self.get_parameter("grid_x_max").value)
            y_min = request.grid_y_min if (request.grid_y_max > request.grid_y_min) else float(self.get_parameter("grid_y_min").value)
            y_max = request.grid_y_max if (request.grid_y_max > request.grid_y_min) else float(self.get_parameter("grid_y_max").value)
            delta_grid = float(self.get_parameter("delta_grid").value)
            nu = float(self.get_parameter("nu").value)
            gamma_context = float(self.get_parameter("gamma_context").value)
            alpha_obs = float(self.get_parameter("alpha_obs").value)
            beta_absent = float(self.get_parameter("beta_absent").value)

            with self._field_lock:
                self._field = GridBayesFilter(
                    x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, delta=delta_grid,
                    nu=nu, gamma_context=gamma_context, alpha_obs=alpha_obs, beta_absent=beta_absent
                )
            self.map_polygon = Polygon([
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max),
            ])

            with self._coverage_mask_lock:
                self._latest_coverage_mask = None

            self.get_logger().info(f"[RESET BELIEF] Target: '{primary}' | Map: '{request.map_name}' | Bounds: [{x_min}, {x_max}, {y_min}, {y_max}] | Grid: {self._field.nx}x{self._field.ny}")

            self._publish_grid_for_rviz()
            self._publish_field()

            response.success = True
            response.message = f"Belief reset successfully for primary_object='{primary}'"
            return response

    # -----------------------------------------------------------------------
    # CORE UPDATE: Update Posterior
    # -----------------------------------------------------------------------
    def update_posterior_callback(self, request, response):
        """Called on demand by the planner. Runs the full Sense step."""
        if self.camera is None:
            response.success = False
            return response

        frame_bgr, pose = self._get_frame_and_pose()
        if frame_bgr is None or pose is None:
            response.success = False
            return response

        # 1. VLM detection
        vlm_detections = self._call_vlm(frame_bgr)

        # 2. Project detections to ground
        grouped_obs, blobs_viz, all_projected_points = self._project_detections(vlm_detections, pose)

        self.get_logger().info(f"Projected {sum(len(v) for v in grouped_obs.values())} valid observations.")

        # 3. Publish detection positions
        self._publish_detections(grouped_obs)

        # 4. Update belief field with Bayes Filter
        stats = self._update_field(pose, grouped_obs, all_projected_points)
        target_found = stats.get("target_found", False) if stats else False

        # Publish target_detected flag to planner
        msg_target      = Bool()
        msg_target.data = target_found
        self.target_detected_pub.publish(msg_target)

        if target_found:
            self.get_logger().info("🔭 TARGET SPOTTED")

        # 5. Publish visualizations
        roi = self._extract_roi(request)
        self._publish_all_viz(frame_bgr, vlm_detections, blobs_viz, roi)

        response.success = True
        response.message = "TARGET_FOUND" if target_found else "OK"
        return response


    def _get_frame_and_pose(self):
        """Returns (frame_bgr, camera pose at that frame's stamp), or (None, None) if unusable."""
        with self._latest_frame_lock:
            frame_bgr = None if self._latest_frame is None else self._latest_frame.copy()
            stamp = self._latest_frame_stamp

        if frame_bgr is None or stamp is None:
            self.get_logger().warn(f"No frame received on {self.rgb_topic} yet.")
            return None, None

        # A stale frame is a stale observation whatever the pose says, so drop it.
        age = (self.get_clock().now() - rclpy.time.Time.from_msg(stamp)).nanoseconds * 1e-9
        if age > self.vlm_frame_delay_tolerance:
            self.get_logger().warn(f"Frame is {age:.2f}s old (> {self.vlm_frame_delay_tolerance:.2f}s) — skipping update.")
            return None, None

        # Fail rather than back-project with the latest pose: on a moving drone that is a ground error.
        pose = lookup_transform_at(self.tf_buffer, self.world_frame, self.camera_frame, stamp, self.tf_timeout_sec)
        if pose is None:
            self.get_logger().warn(f"No {self.world_frame} -> {self.camera_frame} transform at the frame stamp — skipping update.")
            return None, None

        return frame_bgr, pose


    def _call_vlm(self, frame_bgr) -> list:
        vlm_req = DetectSemantics.Request()
        try:
            vlm_req.image = self._bridge.cv2_to_imgmsg(frame_bgr, encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Image encoding failed: {e}")
            return []

        vlm_req.target_query = self._target_query # Unified string for the VLM, e.g. "red car, parking lot, asphalt"

        try:
            self.get_logger().info("VLM is analyzing the frame...")
            vlm_res = self.vlm_client.call(vlm_req)
        except Exception as e:
            self.get_logger().error(f"VLM call failed: {e}")
            return []

        if vlm_res is None or not vlm_res.success:
            return []
        return vlm_res.detections


    def _project_detections(self, vlm_detections, pose):
        """ Projects VLM detections onto the ground plane. Returns (grouped_obs, blobs_viz, all_projected_points)."""
        
        label_to_sigma = dict(self._label_to_sigma)
        label_to_d_star = dict(self._label_to_d_star)

        grouped_obs          = {}
        blobs_viz            = []
        all_projected_points = []

        for det in vlm_detections:
            matched_label = next(
                (key for key in label_to_sigma if key.lower() in det.label.lower()),
                None
            )
            if matched_label is None:
                continue

            mc_res = project_bbox_montecarlo(
                det.x_min, det.y_min, det.x_max, det.y_max, pose, self.camera, self.h_ground
            )
            if mc_res is None:
                continue

            mu, cov, pts2d = mc_res
            all_projected_points.append(pts2d)

            sigma_json = label_to_sigma[matched_label]
            d_star_json = label_to_d_star[matched_label]

            grouped_obs.setdefault(matched_label, []).append((mu, cov, sigma_json, d_star_json))

            # Ellipse params for visualization
            blob = self._make_blob(mu, cov, matched_label)
            blobs_viz.append(blob)

        return grouped_obs, blobs_viz, all_projected_points


    def _make_blob(self, mu, cov, label) -> _BlobViz:
        """Computes ellipse parameters from covariance matrix."""
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx     = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        return _BlobViz(
            x       = float(mu[0]),
            y       = float(mu[1]),
            scale_x = 2.0 * float(np.sqrt(eigvals[0])),
            scale_y = 2.0 * float(np.sqrt(eigvals[1])),
            yaw     = float(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])),
            label   = label,
        )


    def _publish_detections(self, grouped_obs) -> bool:
        label_to_sigma = dict(self._label_to_sigma)

        det_msg                 = DetectionPositions()
        det_msg.header.stamp    = self.get_clock().now().to_msg()
        det_msg.header.frame_id = self.world_frame

        for label, obs_list in grouped_obs.items():
            for mu, cov, sigma_json, d_star_json in obs_list:
                blob = self._make_blob(mu, cov, label)
                det_msg.positions.append(Point(x=float(mu[0]), y=float(mu[1]), z=self.h_ground))
                det_msg.labels.append(label)
                det_msg.radii.append(float(sigma_json))
                det_msg.scale_x.append(blob.scale_x)
                det_msg.scale_y.append(blob.scale_y)
                det_msg.yaw.append(blob.yaw)

        self.detection_positions_pub.publish(det_msg)


    def _update_field(self, pose, grouped_obs, all_projected_points):
        """Updates the belief field and publishes projected points."""
        footprint_mask = None
        fp = compute_ground_footprint(pose, self.camera, self.h_ground, self.max_projection_dist)
        if fp.is_valid and fp.area > 0.1:
            footprint_mask = rasterize_polygon(fp, self._field)
            R_pose = pose[:3, :3]
            body_yaw = float(np.arctan2(R_pose[1, 0], R_pose[0, 0]))
            self._field.record_observation_yaw(footprint_mask, body_yaw)

        with self._coverage_mask_lock:
            coverage_mask = (self._latest_coverage_mask.copy()
                            if self._latest_coverage_mask is not None else None)

        if all_projected_points:
            pts_concat       = np.vstack(all_projected_points)
            pts_xyz          = np.empty((pts_concat.shape[0], 3), dtype=np.float32)
            pts_xyz[:, :2]   = pts_concat
            pts_xyz[:, 2]    = self.h_ground
            self._publish_projected_points(pts_xyz)

        stats = self._field.update_evidence(
            grouped_obs,
            footprint_mask=footprint_mask,
            coverage_mask=coverage_mask,
        )
        if stats and stats["alpha_stats"]:
            pi_0 = 1.0 / float(self._field.nx * self._field.ny)
            pi_min = stats.get("pi_min", float(np.min(self._field.pi)))
            pi_max = stats.get("pi_max", float(np.max(self._field.pi)))
            pi_mean = float(np.mean(self._field.pi))
            
            self.get_logger().info(f"📊 Posterior: pi_0={pi_0:.8f}, pi_min={pi_min:.8f}, pi_max={pi_max:.8f}, pi_mean={pi_mean:.8f}")
            for lbl, (a_min, a_max) in stats["alpha_stats"].items():
                l_single = stats["label_l_max"].get(lbl, 1.0)
                self.get_logger().info(f"   └─ '{lbl}': L_single={l_single:.4f}, L_pos={stats['L_pos_max']:.4f}")
        return stats

    def _extract_roi(self, request):
        """Extracts ROI from request if valid."""
        if request.roi_x_max > request.roi_x_min and request.roi_y_max > request.roi_y_min:
            return (request.roi_x_min, request.roi_x_max, request.roi_y_min, request.roi_y_max)
        return None

    def evaluate_waypoints_callback(self, request: EvaluateWaypoints.Request, response: EvaluateWaypoints.Response) -> EvaluateWaypoints.Response:
        candidates = request.candidates

        if not candidates or self.camera is None or self.T_body_to_opt is None:
            response.success = False
            response.message = "Not ready: candidates, camera intrinsics, or static TF not loaded yet"
            self.get_logger().warn(response.message)
            return response

        # Unpack service request
        cx         = float(request.current_x)  # current drone's x
        cy         = float(request.current_y)  # current drone's y
        gamma      = float(request.gamma)      # distance discount factor
        w_semantic = float(request.w_semantic) # semantic gain weight factor
        w_reobs    = float(request.w_reobs)    # reobservation gain weight factor
        
        altitude = self.current_position.z 

        pi = self._field.get_posterior_probability()
        ox, oy, res = self._field.x_min, self._field.y_min, self._field.delta
        nx, ny = self._field.nx, self._field.ny
        pi_0 = self._field.pi_0

        # Build active domain corona mask D_t = buffer(C_t, R_active) \ C_t
        active_domain_radius = self.get_parameter("active_domain_radius").value
        kernel_size = max(1, int(np.ceil(active_domain_radius / res)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kernel_size + 1, 2 * kernel_size + 1))

        consumed_uint8 = self._field.consumed.astype(np.uint8)
        if consumed_uint8.any():
            dilated_consumed = cv2.dilate(consumed_uint8, kernel).astype(bool)
            active_domain_mask = dilated_consumed & (~self._field.consumed)
        else:
            active_domain_mask = ~self._field.consumed

        scores = []
        ig_geometrics = []
        ig_semantics = []
        ig_reobservations = []
        best_J = -1e12
        best_idx = -1

        for idx, cand in enumerate(candidates):
            x, y, yaw = float(cand.x), float(cand.y), float(cand.theta)
            dist = float(np.hypot(x - cx, y - cy))
            pose_w = build_candidate_pose(x, y, yaw, altitude, self.T_body_to_opt)
            fp = compute_ground_footprint(pose_w, self.camera, self.h_ground, self.max_projection_dist)
            roi_sem = fp.intersection(self.map_polygon)


            if not fp.is_valid or fp.is_empty or roi_sem.is_empty or roi_sem.area <= 0.0:
                scores.append(-1e12)
                ig_geometrics.append(0.0)
                ig_semantics.append(0.0)
                ig_reobservations.append(0.0)
                continue

            xmin, ymin, xmax, ymax = roi_sem.bounds
            j0 = max(0, int((xmin - ox) / res))
            j1 = min(nx - 1, int((xmax - ox) / res))
            i0 = max(0, int((ymin - oy) / res))
            i1 = min(ny - 1, int((ymax - oy) / res))

            if j1 < j0 or i1 < i0:
                ig_sem = ig_geom = ig_reobs = 0.0
            else:
                js = np.arange(j0, j1 + 1)
                is_ = np.arange(i0, i1 + 1)
                CX, CY = np.meshgrid(ox + (js + 0.5) * res, oy + (is_ + 0.5) * res)

                prepare(roi_sem)
                mask = contains_xy(roi_sem, CX.ravel(), CY.ravel()).reshape(CX.shape)

                pi_slice = pi[i0:i1+1, j0:j1+1]
                active_domain_slice = active_domain_mask[i0:i1+1, j0:j1+1]
                consumed_slice = self._field.consumed[i0:i1+1, j0:j1+1]
                new_cells_mask = mask & active_domain_slice

                # Probabilistic formulation (terms are probability masses)
                # Geometric IG: prior probability of newly discovered cells
                ig_geom = float(np.sum(new_cells_mask)) * pi_0 # TODO: this is wrong because it has to account for the cells in the posterior too (it may extend outside the current footprint)

                # Semantic IG: excess probability in the entire footprint mask (allows re-observing clues)
                excess_prob = np.maximum(pi_slice - pi_0, 0.0)
                ig_sem = float(np.sum(excess_prob[new_cells_mask])) * w_semantic

                # Re-observation IG: reward observing already covered cells from novel yaw angles
                ig_reobs = self._field.compute_reobservation_gain(yaw, mask, pi_slice, consumed_slice, i0, i1, j0, j1) * w_reobs 

                # Total IG is the weighted sum of terms
                ig_total = ig_geom + ig_sem + ig_reobs

            # Candidate Objective J(w) = I(w) - gamma * distance
            J = ig_total - gamma * dist

            #self.get_logger().info(f"Candidate {idx}: J={J:.4f}, ig_geom={ig_geom:.4f}, ig_sem={ig_sem:.4f}, ig_reobs={ig_reobs:.4f}, pensalty={gamma*dist:.4f}")

            scores.append(J)
            ig_geometrics.append(ig_geom)
            ig_semantics.append(ig_sem)
            ig_reobservations.append(ig_reobs)

            if J > best_J:
                best_J = J
                best_idx = idx

        response.success          = True
        response.message          = f"Evaluated {len(candidates)} waypoints. Best index: {best_idx}"
        response.best_index       = best_idx
        response.best_waypoint    = candidates[best_idx]
        response.scores           = scores
        response.ig_geometric     = ig_geometrics
        response.ig_semantic      = ig_semantics
        response.ig_reobservation = ig_reobservations
        return response
    
    # -----------------------------------------------------------------------
    # PUBLISHERS
    # -----------------------------------------------------------------------
    
    def _publish_all_viz(self, frame_bgr, vlm_detections, blobs_viz, roi):
        label_to_sigma = dict(self._label_to_sigma)

        self._publish_annotated_bboxes(frame_bgr, vlm_detections, label_to_sigma)
        self._publish_field(roi=roi)
        self._publish_grid_for_rviz()
        self._publish_blob_markers(blobs_viz, label_to_sigma)
        self._publish_target_estimate()

    def _publish_field(self, roi=None):
        """Publishes the posterior field as a PosteriorFieldMsg over the full map grid."""
        with self._field_lock:
            pi = self._field.get_posterior_probability()
            origin_x = self._field.x_min
            origin_y = self._field.y_min

            msg = PosteriorFieldMsg()
            msg.origin_x   = float(origin_x)
            msg.origin_y   = float(origin_y)
            msg.resolution = float(self._field.delta)
            msg.width      = int(pi.shape[1])
            msg.height     = int(pi.shape[0])
            msg.data       = pi.flatten().astype(np.float32).tolist()
            self.field_pub.publish(msg)
        
    def _publish_annotated_bboxes(self, frame_bgr, detections, label_dict):
        """Publishes the annotated image with bounding boxes and labels."""
        annotated = frame_bgr.copy()
        for det in detections:
            x1, y1 = int(det.x_min), int(det.y_min)
            x2, y2 = int(det.x_max), int(det.y_max)
            label = det.label
            
            # Identify if the detection is target (red) or a hint (green)
            is_target = any((k.lower() in det.label.lower() and label_dict[k] == 0.0) for k in label_dict)
            color = (0, 0, 255) if is_target else (0, 255, 0)
            
            # Draw the bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Compute text dimensions
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # If bbox is in the upper part of the image, draw the text box inside
            bg_y1 = y1 - text_h - 6
            bg_y2 = y1
            txt_y = y1 - 5
            
            if bg_y1 < 0:
                bg_y1 = y1
                bg_y2 = y1 + text_h + 6
                txt_y = y1 + text_h + 1

            # Draw the background rectangle for text
            cv2.rectangle(annotated, (x1, bg_y1), (x1 + text_w, bg_y2), color, -1)

            # Draw the label text
            cv2.putText(annotated, label, (x1, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        try:
            msg = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.camera_frame
            self.annotated_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish annotated image: {e}")

    def _publish_grid_for_rviz(self):
        """Publishes the posterior belief peaks formatted for RViz Costmap visualization"""
        with self._field_lock:
            pi = self._field.get_posterior_probability()
            if pi.size == 0:
                return

            pi_0 = 1.0 / float(pi.size)
            pi_max = float(np.max(pi))

            if (pi_max - pi_0) < 1e-12:
                # Nessun indizio positivo rilevato -> mappa completamente pulita (0)
                data_int = np.zeros(pi.shape, dtype=np.int8)
            else:
                # Mappa solo i picchi positivi sopra la prior da 0 a 100
                positive_delta = np.maximum(pi - pi_0, 0.0)
                data_int = np.clip(np.round((positive_delta / (pi_max - pi_0)) * 100.0), 0, 100).astype(np.int8)

            og = OccupancyGrid()
            og.header.frame_id = self.world_frame
            og.header.stamp = self.get_clock().now().to_msg()
            og.info.resolution = float(self._field.delta)
            og.info.width = int(self._field.nx)
            og.info.height = int(self._field.ny)
            og.info.origin = Pose()
            og.info.origin.position.x = float(self._field.x_min)
            og.info.origin.position.y = float(self._field.y_min)
            og.info.origin.position.z = float(self.h_ground + 0.02)
            og.info.origin.orientation.w = 1.0
            og.data = data_int.flatten().tolist()
            self.grid_pub.publish(og)

    def _publish_blob_markers(self, blobs: list[_BlobViz], label_dict):
        """Publishes the detected blobs as MarkerArray for RViz visualization."""
        ma = MarkerArray()
        clear = Marker() # erase old markers
        clear.header.frame_id = self.world_frame
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        z = self.h_ground + 0.06
        for k, blob in enumerate(blobs):
            # Convert the yaw angle to a quaternion for rviz
            qw = float(np.cos(blob.yaw / 2.0))
            qz = float(np.sin(blob.yaw / 2.0))

            # Internal marker (internal ellipse)
            m_inner = Marker()
            m_inner.header.frame_id = self.world_frame
            m_inner.header.stamp = self.get_clock().now().to_msg()
            m_inner.ns = "geo_footprint"
            m_inner.id = k * 2
            m_inner.type = Marker.CYLINDER
            m_inner.pose.position.x, m_inner.pose.position.y, m_inner.pose.position.z = blob.x, blob.y, z
            
            # Apply rotation and correct scale
            m_inner.pose.orientation.x = 0.0
            m_inner.pose.orientation.y = 0.0
            m_inner.pose.orientation.z = qz
            m_inner.pose.orientation.w = qw
            m_inner.scale.x = blob.scale_x
            m_inner.scale.y = blob.scale_y
            m_inner.scale.z = 0.1
            
            m_inner.color.r, m_inner.color.g, m_inner.color.b, m_inner.color.a = 0.0, 1.0, 0.0, 0.8
            ma.markers.append(m_inner)

            # --- External marker (isotropically expanded Flat-Top sigma_c) ---
            sigma_c = label_dict.get(blob.label, 0.0)
            if sigma_c > 0:
                m_outer = Marker()
                m_outer.header.frame_id = self.world_frame
                m_outer.header.stamp = self.get_clock().now().to_msg()
                m_outer.ns = "flattop_radius"
                m_outer.id = k * 2 + 1
                m_outer.type = Marker.CYLINDER
                m_outer.pose.position.x, m_outer.pose.position.y, m_outer.pose.position.z = blob.x, blob.y, z - 0.01
                
                # Follow the same rotation, but expand by sigma_c on all sides
                m_outer.pose.orientation.z = qz
                m_outer.pose.orientation.w = qw
                m_outer.scale.x = blob.scale_x + (2.0 * sigma_c)
                m_outer.scale.y = blob.scale_y + (2.0 * sigma_c)
                m_outer.scale.z = 0.05
                
                m_outer.color.r, m_outer.color.g, m_outer.color.b, m_outer.color.a = 1.0, 1.0, 0.0, 0.2
                ma.markers.append(m_outer)
                
        self.blobs_pub.publish(ma)

    def _publish_target_estimate(self):
        pi = self._field.get_posterior_probability() 
        if pi.size == 0:
            return
            
        pi_max = float(np.max(pi))
        pi_0 = float(self._field.pi_0)

        # Publish only if there's a probability peak (e.g. at least 10% higher than background noise)
        if pi_max < pi_0 * 1.1: 
            return

        # Find the indices of the maximum probability 
        idx = np.unravel_index(np.argmax(pi), pi.shape)
        x_est, y_est = self._field.idx_to_world(idx[0], idx[1])

        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "target_estimate"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(x_est)
        m.pose.position.y = float(y_est)
        m.pose.position.z = float(self.h_ground + 0.5)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 5.0 * self.marker_scale
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 1.0
        
        self.target_estimate_pub.publish(m)

    def _publish_projected_points(self, pts_xyz: np.ndarray):
        """Publishes ground-projected sample points as PointCloud2 for RViz debugging."""
        if pts_xyz.size == 0:
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.world_frame

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        pts = np.ascontiguousarray(pts_xyz, dtype=np.float32)
        n_points = pts.shape[0]

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = n_points
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * n_points
        msg.is_dense = True
        msg.data = pts.tobytes()

        self.projected_points_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()