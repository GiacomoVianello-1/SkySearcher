import threading
import numpy as np
import cv2
import json

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from cv_bridge import CvBridge

from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import Bool, Header

from interfaces.msg import PosteriorField as PosteriorFieldMsg, CoverageField, DetectionPositions
from interfaces.srv import DetectSemantics, UpdatePosterior

from shapely.geometry import Polygon, MultiPolygon, Point as ShapelyPoint
from tf2_ros import Buffer, TransformListener

from exploration.camera import Camera
from exploration.utils import lookup_camera_pose, compute_ground_footprint
from exploration.active_domain import rasterize_polygon

from dataclasses import dataclass

@dataclass
class _BlobViz:
    """Minimal blob for RViz markers visualization."""
    x: float        # X coordinate of the blob center
    y: float        # Y coordinate of the blob center
    scale_x: float  # Length of the major axis (ellipse)
    scale_y: float  # Length of the minor axis (ellipse)
    yaw: float      # Rotation angle of the ellipse
    q: float        # Quality or confidence (0.0 to 1.0) for visualization purposes
    label: str      # Label of the blob (for debugging/visualization)


# ===========================================================================
# CORE MATHS: Flat-Top Belief Tracker (TODO: move it to a separate file)
# ===========================================================================

class FlatTopBeliefTracker:
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float, delta: float,
                p_0: float = 0.05, p_min: float = 0.001, p_max: float = 0.9999,
                l_neg: float = -0.3, k_pos: float = 2.0, nu: float = 0.5,  # METRES — exponential decay scale outside the semantic buffer
                p_max_context: float = 0.85,
                coverage_attenuation: float = 0.1):  # multiplicative factor for kernel mass falling on already-observed ground
        
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.y_min, self.y_max = float(y_min), float(y_max)
        self.delta = float(delta)

        self.nx = int(np.ceil((x_max - x_min) / delta))
        self.ny = int(np.ceil((y_max - y_min) / delta))

        # Log-Odds bounds
        self.l_0 = np.log(p_0 / (1.0 - p_0))
        self.l_min = np.log(p_min / (1.0 - p_min))
        self.l_max = np.log(p_max / (1.0 - p_max))

        # Structural Clamping ceiling: maximum log-odds reachable through contextual
        # hints alone. Only direct target detections are allowed to push the belief
        # above this threshold, preserving the "shatter the ceiling" semantics of
        # Equation (robust_update) in the paper.
        self.l_max_context = np.log(p_max_context / (1.0 - p_max_context))
        
        self.l_neg = float(l_neg)
        self.k_pos = float(k_pos)
        self.nu = float(nu) # External blur of the Flat-Top Kernel
        self.coverage_attenuation = float(coverage_attenuation)

        # Map state
        self.log_odds = np.full((self.ny, self.nx), self.l_0, dtype=np.float64)
        self.max_evidence_history = {}
        self.consumed = np.zeros((self.ny, self.nx), dtype=bool)

        _xs = self.x_min + (np.arange(self.nx) + 0.5) * self.delta
        _ys = self.y_min + (np.arange(self.ny) + 0.5) * self.delta
        self.XX, self.YY = np.meshgrid(_xs, _ys)

    def _compute_mahalanobis(self, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Computes the Mahalanobis distance from the Gaussian defined by (mu, cov) to each cell center."""
        pos = np.dstack((self.XX, self.YY))
        diff = pos - mu
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.eye(2)
        D_sq = np.einsum('...i,ij,...j->...', diff, inv_cov, diff) # Compact form for computing the Mahalanobis distance
        return np.sqrt(np.maximum(0, D_sq))

    def _euclid_distance_from_ellipse(self, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Euclidean distance (in METERS) from each grid cell centre to the boundary
        of the 1-sigma ellipse defined by (mu, cov). Cells inside the ellipse
        return 0.

        Approximation: along the ray (mu -> x), the ellipse boundary lies at
        euclidean fraction 1/D_M of the way out, so the surface distance is
            d_E(x) ~= ||x - mu|| * (1 - 1/D_M)
        when D_M >= 1, and 0 otherwise.

        This is exact for isotropic (circular) ellipses and a high-quality
        first-order approximation for moderately anisotropic ones. It keeps the
        semantic radius sigma_k commensurable with d_E in the kernel formula:
        both are now in metres.
        """
        D_M = self._compute_mahalanobis(mu, cov)
        diff = np.dstack((self.XX - mu[0], self.YY - mu[1]))
        d_centre = np.linalg.norm(diff, axis=-1)
        D_M_safe = np.maximum(D_M, 1.0 + 1e-9)
        d_surface = d_centre * (1.0 - 1.0 / D_M_safe)
        return np.where(D_M <= 1.0, 0.0, d_surface)

    def update_evidence(self, grouped_observations: dict, footprint_mask: np.ndarray = None, coverage_mask: np.ndarray = None):
        """
        Applies Max-Fusion + Flat-Top Kernel for contexts, with Structural Clamping
        on contextual contributions. The target contribution bypasses the clamp.

        grouped_observations format: {label: [(mu, cov, sigma_json), ...]}

        coverage_mask : boolean array of shape (ny, nx), True on cells that
            have already been physically observed by the drone (cumulative
            ground footprint). On those cells the freshly computed kernel
            value S_c is multiplied by self.coverage_attenuation (< 1),
            because direct visual evidence overrides indirect contextual
            evidence: if the target/hint were really there, we would have
            seen the target geometrically. We attenuate S_c rather than the
            stored log-odds, so past contributions accumulated outside the
            then-current coverage are preserved; only *new* mass added on
            already-seen ground is downweighted.
        """
        S_global_current = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Target-only kernel accumulator. Used to gate the empty-mask in
        # step 4: a cell inside the current footprint is "observed empty"
        # iff no target Gaussian peaks over it. Contextual hints (Flat-Top)
        # must NOT protect the cell from the hard reset — they're indirect
        # evidence by construction and routinely sprawl over the footprint
        # without saying anything about target presence under the camera.
        # Computed pre-coverage-attenuation so that target memory in
        # already-observed regions is still protected from cancellation.
        S_global_target = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Separate accumulators: contextual hints are subject to the clamp,
        # target detections are not.
        delta_context = np.zeros((self.ny, self.nx), dtype=np.float64)
        delta_target = np.zeros((self.ny, self.nx), dtype=np.float64)

        # 1. Positive Semantic Update
        for label, obs_list in grouped_observations.items():
            if not obs_list:
                continue

            sigma_k_meters = float(obs_list[0][2])
            is_target = (sigma_k_meters == 0.0)

            if is_target:
                # Target: pure Gaussian on Mahalanobis distance.
                D_list = [self._compute_mahalanobis(mu, cov) for mu, cov, _ in obs_list]
                D_min = np.min(D_list, axis=0)
                S_c = np.exp(-(D_min**2) / 2.0)
                # Track target footprint BEFORE coverage attenuation so a
                # re-detection over already-observed ground still protects
                # the cell from the empty-mask hard reset.
                S_global_target = np.maximum(S_global_target, S_c)
            else:
                # Hint: Flat-Top Kernel.
                d_list = [self._euclid_distance_from_ellipse(mu, cov)
                        for mu, cov, _ in obs_list]
                d_min = np.min(d_list, axis=0)
                excess = np.maximum(0.0, d_min - sigma_k_meters)
                S_c = np.exp(-(excess**2) / (2.0 * self.nu**2))

            # Max-Fusion
            if label not in self.max_evidence_history:
                self.max_evidence_history[label] = np.zeros((self.ny, self.nx), dtype=np.float64)
            storico = self.max_evidence_history[label]
            delta_S = np.maximum(0.0, S_c - storico)
            self.max_evidence_history[label] = np.maximum(storico, S_c)

            # ---- Coverage attenuation -------------------------------------
            if coverage_mask is not None:
                S_c = np.where(coverage_mask, S_c * self.coverage_attenuation, S_c)
            # ---------------------------------------------------------------

            S_global_current = np.maximum(S_global_current, S_c)

            if is_target:
                gain = self.k_pos * 6.0
                delta_target += (gain * delta_S)
            else:
                delta_context += (self.k_pos * 1.0 * delta_S)

        # 2. Structural Clamping on contextual contributions
        c_avail = np.maximum(0.0, self.l_max_context - self.log_odds)
        u_context = np.minimum(c_avail, delta_context)

        # 3. Apply updates: clamped context + unclamped target
        self.log_odds += u_context
        self.log_odds += delta_target

        nonzero = (self.log_odds > self.l_0 + 1e-6).sum()
        print(f"[update_evidence] celle sopra prior: {nonzero}")
        print(f"l_0={self.l_0:.4f}, l_max_context={self.l_max_context:.4f}")

        if footprint_mask is not None:
            empty_mask = footprint_mask & (S_global_target < 0.1)
            self.log_odds[empty_mask] = self.l_0
            for label in self.max_evidence_history:
                self.max_evidence_history[label][empty_mask] = 0.0

        # 5. Global safety clamp (numerical stability only)
        self.log_odds = np.clip(self.log_odds, self.l_min, self.l_max)

        return S_global_current

    def get_posterior_probability(self) -> np.ndarray:
        """Converts the log-odds to probability and applies the consumed mask."""
        P = np.exp(self.log_odds) / (1.0 + np.exp(self.log_odds))
        P[self.consumed] = 0.0
        return P
    
    def get_log_F(self) -> np.ndarray:
        """Returns the raw Log-Odds map with consumed cells set to -inf."""
        out = self.log_odds.copy()
        out[self.consumed] = -np.inf
        return out

    def idx_to_world(self, i: int, j: int):
        """Converts grid indices to world coordinates (center of the cell)."""
        return (self.x_min + (j + 0.5) * self.delta, self.y_min + (i + 0.5) * self.delta)

    def reset(self):
        """Resets the belief tracker to the initial state."""
        self.log_odds.fill(self.l_0)
        self.max_evidence_history.clear()
        self.consumed.fill(False)


# ===========================================================================
# ROS NODE: Semantic Map Node (Flat-Top Variant)
# ===========================================================================

class SemanticMapNode(Node):
    def __init__(self):
        super().__init__("semantic_map_node")

        # --- PARAMETERS ---
        self.declare_parameter("camera_frame",      "camera_color_optical_frame")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("rgb_topic",         "/camera/camera/color/image_raw")

        self.declare_parameter("h_ground", 0.0) # Height of the ground plane for ray-casting (flat earth assumption)
        self.declare_parameter("max_projection_dist", 200.0)

        # Grid map
        self.declare_parameter("grid_x_min", -10.0)
        self.declare_parameter("grid_x_max",  10.0)
        self.declare_parameter("grid_y_min", -10.0)
        self.declare_parameter("grid_y_max",  10.0)
        self.declare_parameter("delta_grid",   0.1)    # Resolution of the grid map (meters per cell)

        self.declare_parameter("instruction", "rowing boat in the lake") # NL instruction for the VLM
        self.declare_parameter("vlm_log_path", "vlm_scale_clues_log.json") # Where is the fixture JSON for label->sigma mapping
        self.declare_parameter("default_sigma", 25.0) # Default sigma for labels not in the fixture (meters, in world space)
        self.declare_parameter("coverage_attenuation", 0.1) # Multiplicative factor applied to kernel mass falling on already-observed ground (cumulative coverage). 0 = hard mask, 1 = no attenuation (legacy behaviour).

        # Resolve parameters
        self.camera_frame = self.get_parameter("camera_frame").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        rgb_topic = self.get_parameter("rgb_topic").value

        self.h_ground = float(self.get_parameter("h_ground").value)
        self.max_projection_dist = float(self.get_parameter("max_projection_dist").value)
        self.default_sigma = float(self.get_parameter("default_sigma").value)
        coverage_attenuation = float(self.get_parameter("coverage_attenuation").value)

        x_min = float(self.get_parameter("grid_x_min").value)
        x_max = float(self.get_parameter("grid_x_max").value)
        y_min = float(self.get_parameter("grid_y_min").value)
        y_max = float(self.get_parameter("grid_y_max").value)
        delta_grid = float(self.get_parameter("delta_grid").value)

        self._bridge = CvBridge()
        self.cb_group = ReentrantCallbackGroup()

        # --- INITIALIZATION OF THE MATHEMATICAL TRACKER ---
        self._field = FlatTopBeliefTracker(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, delta=delta_grid, coverage_attenuation=coverage_attenuation)
        self.get_logger().info(f"Posterior Flat-Top Grid Initialized ({self._field.nx}x{self._field.ny}), coverage_attenuation={coverage_attenuation}")

        # --- INTERNAL STATE ---
        self.camera = None
        self._latest_frame = None
        self._latest_frame_lock = threading.Lock()

        self._prompts_lock = threading.Lock()
        self._instruction = ""
        self._primary_object = ""
        self._label_to_sigma = {}  # label -> sigma(json) mapping
        self._target_query = ""    # Unified string for the VLM
        self._latest_coverage_mask: np.ndarray | None = None
        self._coverage_mask_lock = threading.Lock()

        self._last_vlm_inference_time = 0.0 # Used by the planner for filling the feedback message

        # --- TF AND SUBSCRIBERS ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(Image, rgb_topic, self.image_callback, 5)
        self.create_subscription(CoverageField, "/coverage_field", self._coverage_field_cb, 1)

        # --- CLIENT AND SERVICES ---
        # Connection to vlm_node
        self.vlm_client = self.create_client(DetectSemantics, 'detect_semantic_clues', callback_group=self.cb_group)
        self.update_srv = self.create_service(UpdatePosterior, "/update_posterior", self.update_posterior_callback, callback_group=self.cb_group)

        # --- PUBLISHERS ---
        self.acquired_img_pub = self.create_publisher(Image, "acquired_img", 10)
        self.annotated_pub = self.create_publisher(Image, "/annotated_img", 5)
        self.field_pub = self.create_publisher(PosteriorFieldMsg, "/posterior_field", 1)
        self.blobs_pub = self.create_publisher(MarkerArray, "/semantic_blobs", 5)
        self.target_estimate_pub = self.create_publisher(Marker, "/target_estimate_marker", 1)
        self.grid_pub = self.create_publisher(OccupancyGrid, "/posterior_grid", 1)
        self.projected_points_pub = self.create_publisher(PointCloud2, "/projected_bbox_points", 5)
        self.target_detected_pub = self.create_publisher(Bool, "/target_detected", 10)
        self.detection_positions_pub = self.create_publisher(DetectionPositions, "/detection_positions", 1)

        self._load_prompts_from_json()
        self.get_logger().info("Semantic Map Node (Flat-Top) ready.")

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def _load_prompts_from_json(self) -> bool:
        instruction = self.get_parameter("instruction").value.strip()
        vlm_log_path = self.get_parameter("vlm_log_path").value

        if not instruction:
            self.get_logger().error("Parameter 'instruction' is empty. Cannot load prompts.")
            return False

        try:
            with open(vlm_log_path, 'r') as f:
                vlm_logs = json.load(f)
        except Exception as e:
            self.get_logger().error(f"Failed to load VLM log file '{vlm_log_path}': {e}")
            return False

        mission_data = None
        for entry in vlm_logs:
            if str(entry.get("instruction", "")).strip() == instruction:
                mission_data = entry["response"]
                break

        if mission_data is None:
            self.get_logger().error(f"Instruction '{instruction}' not found in '{vlm_log_path}'.")
            return False

        primary     = mission_data["primary_object"]
        surroundings = mission_data["surroundings"]
        sigmas      = mission_data["sigmas"]

        with self._prompts_lock:
            self._instruction     = instruction
            self._primary_object  = primary
            self._label_to_sigma  = {primary: 0.0}
            for s, sig in zip(surroundings, sigmas):
                self._label_to_sigma[s.strip()] = float(sig)
            self._target_query = ", ".join([primary] + surroundings)

        self._field.reset()
        self.get_logger().info(
            f"Prompts loaded: instruction='{instruction}', primary='{primary}', "
            f"surroundings={surroundings}, sigmas={sigmas}"
        )
        return True

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------
    def camera_info_callback(self, msg: CameraInfo):
        if self.camera is None:
            self.camera = Camera(
                h=msg.height,
                w=msg.width,
                fx=msg.k[0],   # K[0,0]
                fy=msg.k[4],   # K[1,1]
                cx=msg.k[2],   # K[0,2]
                cy=msg.k[5],   # K[1,2]
                depth=self.max_projection_dist
            )
            self.get_logger().info(
                f"Camera model initialised from CameraInfo: "
                f"{msg.width}x{msg.height}, "
                f"fx={msg.k[0]:.2f}, fy={msg.k[4]:.2f}, "
                f"cx={msg.k[2]:.2f}, cy={msg.k[5]:.2f}"
            )
 
            # Warn if the stream carries non-zero distortion coefficients.
            # The Camera class and the Monte Carlo projector do NOT apply
            # undistortion — use use_rectified:=true for pixel-accurate results.
            if msg.d and any(abs(d) > 1e-6 for d in msg.d):
                self.get_logger().warn(
                    f"Non-zero distortion coefficients detected: {list(msg.d)}. "
                    "The projection pipeline does NOT undistort pixels. "
                    "Switch to use_rectified:=true or subscribe to the "
                    "image_rect_color stream for metric accuracy."
                )

    def image_callback(self, msg: Image):
        with self._latest_frame_lock:
            self._latest_frame = msg
    
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

    # Ray-Casting Monte Carlo
    def _project_bbox_montecarlo(self, xmin, ymin, xmax, ymax, pose, n_samples=1000):
        """
        Projects the 2D bounding box onto the ground plane using Monte Carlo ray-casting.
        Samples points within the bbox, casts rays from the camera, and finds their intersection with the ground plane.
        Returns the geometric mean and covariance of the intersection points in world coordinates.
        """
        rng = np.random.default_rng()
        u = rng.uniform(xmin, xmax, size=n_samples)
        v = rng.uniform(ymin, ymax, size=n_samples)

        # Pinhole camera model: convert pixel coordinates to normalized camera coordinates
        x_opt = (u - self.camera.cx) / self.camera.fx
        y_opt = (v - self.camera.cy) / self.camera.fy
        z_opt = np.ones_like(x_opt)
        
        rays_opt = np.stack([x_opt, y_opt, z_opt], axis=0)
        R, t = pose[:3, :3], pose[:3, 3]
        rays_world = R.dot(rays_opt)

        ray_z = rays_world[2, :]
        lambda_intersect = (self.h_ground - t[2]) / ray_z

        # Valid Rays: point downward, have a minimum slope to the ground
        min_slope = np.sin(np.deg2rad(3.0))
        valid_mask = (lambda_intersect > 0) & (ray_z < -min_slope)
        
        if np.sum(valid_mask) < 10:
            return None

        pts_world = t[:, np.newaxis] + rays_world[:, valid_mask] * lambda_intersect[valid_mask]

        # Distance Filter: keep only points within a reasonable radius from the camera (e.g., 200m) to discard outliers due to numerical issues
        dist_sq = (pts_world[0, :] - t[0])**2 + (pts_world[1, :] - t[1])**2
        pts_final = pts_world[:2, dist_sq < (200.0 ** 2)]

        if pts_final.shape[1] < 10:
            return None

        mean_xy = pts_final.mean(axis=1)

        # Addictive Isotropic Floor (regularization + minimum uncertainty of the detector)
        sigma_min = 0.5  # meters — minimum uncertainty due to pixel quantization + bbox jitter
        cov = np.cov(pts_final) + (sigma_min**2) * np.eye(2)

        # Upper Cap ONLY as safety net (proportional, preserves the shape)
        eigvals, eigvecs = np.linalg.eigh(cov)
        sigma_max = 50.0
        if eigvals.max() > sigma_max**2:
            eigvals = eigvals * (sigma_max**2 / eigvals.max())
            cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return mean_xy, cov, pts_final.T

    # -----------------------------------------------------------------------
    
    # MAIN LOOP: Update Posterior (SENSE PLANE ACT)

    def update_posterior_callback(self, request, response):
        """Called on demand by the planner. Runs the full Sense step."""
        if self.camera is None:
            response.success = False
            return response

        frame_ros, pose = self._get_frame_and_pose()
        if frame_ros is None or pose is None:
            response.success = False
            return response
        
        self.acquired_img_pub.publish(frame_ros) # Publish the acquired image for debugging/visualization. 

        # 1. VLM detection
        vlm_detections = self._call_vlm(frame_ros)
        response.vlm_inference_time_sec = self._last_vlm_inference_time

        # 2. Project detections to ground
        grouped_obs, blobs_viz, all_projected_points = self._project_detections(vlm_detections, pose)

        self.get_logger().info(f"Projected {sum(len(v) for v in grouped_obs.values())} valid observations.")

        # 3. Publish detection positions and target flag
        target_found = self._publish_detections(grouped_obs)

        # 4. Update belief field
        self._update_field(pose, grouped_obs, all_projected_points)

        # Populate response fields
        F = self._field.get_posterior_probability()
        response.max_posterior = float(F.max())

        # 5. Publish visualizations
        roi = self._extract_roi(request)
        frame_bgr = self._bridge.imgmsg_to_cv2(frame_ros, desired_encoding="bgr8")
        self._publish_all_viz(frame_bgr, vlm_detections, blobs_viz, roi)

        if target_found:
            # Estrai la stima della posizione del target come media pesata
            # della gaussiana proiettata nell'ultimo step
            with self._prompts_lock:
                label_to_sigma = dict(self._label_to_sigma)

            target_mus = [
                mu for label, obs_list in grouped_obs.items()
                if label_to_sigma.get(label, -1) == 0.0
                for mu, cov, _ in obs_list
            ]

            if target_mus:
                mean_mu = np.mean(target_mus, axis=0)
                response.target_estimate_x = float(mean_mu[0])
                response.target_estimate_y = float(mean_mu[1])

        response.success = True
        response.message = "TARGET_FOUND" if target_found else "OK"
        return response

    # TODO: adjust to Jetson (see vlm_ros/query_node.py for reference)
    def _get_frame_and_pose(self):
        """Returns (frame_ros, pose) from AirSim or (None, None) if not available."""
        with self._latest_frame_lock:
            frame_ros = self._latest_frame
        pose = lookup_camera_pose(self.tf_buffer, "world", self.camera_frame) # from utils.py
        return frame_ros, pose

    def _call_vlm(self, frame_ros) -> list:
        """Calls the VLM service on the acquired frame. Returns list of detections (empty on failure)."""
        with self._prompts_lock:
            target_query      = self._target_query
            label_to_sigma    = dict(self._label_to_sigma)

        vlm_req = DetectSemantics.Request()
        vlm_req.image = frame_ros           # The latest acquired image
        vlm_req.target_query = target_query # Unified string for the VLM, e.g. "red car, parking lot, asphalt"

        try:
            self.get_logger().info("VLM is analyzing the frame...")
            vlm_res = self.vlm_client.call(vlm_req)
        except Exception as e:
            self.get_logger().error(f"VLM call failed: {e}")
            return []

        if vlm_res is None:
            return []
        
        self._last_vlm_inference_time = float(vlm_res.inference_time_sec)

        if not vlm_res.success:
            return []
    
        return vlm_res.detections


    def _project_detections(self, vlm_detections, pose):
        """ Projects VLM detections onto the ground plane. Returns (grouped_obs, blobs_viz, all_projected_points)."""
        with self._prompts_lock:
            label_to_sigma = dict(self._label_to_sigma)

        grouped_obs          = {} # label -> list of (mu, cov, sigma_json) tuples for the current detections, used for belief update
        blobs_viz            = [] # List of _BlobViz for RViz visualization of the detections as ellipses
        all_projected_points = [] # List of all projected 2D points (in world coordinates) from all detections, used for PointCloud visualization

        for det in vlm_detections:
            matched_label = next(
                (key for key in label_to_sigma if key.lower() in det.label.lower()),
                None
            )
            if matched_label is None:
                continue

            mc_res = self._project_bbox_montecarlo(det.x_min, det.y_min, det.x_max, det.y_max, pose)
            if mc_res is None:
                continue

            mu, cov, pts2d = mc_res
            all_projected_points.append(pts2d)
            sigma_json = label_to_sigma[matched_label]

            grouped_obs.setdefault(matched_label, []).append((mu, cov, sigma_json))

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
            q       = 1.0,
            label   = label,
        )


    def _publish_detections(self, grouped_obs) -> bool:
        """
        Publishes /detection_positions and /target_detected.
        Returns True if target was found in this step.
        """
        with self._prompts_lock:
            label_to_sigma = dict(self._label_to_sigma)

        det_msg                 = DetectionPositions()
        det_msg.header.stamp    = self.get_clock().now().to_msg()
        det_msg.header.frame_id = "world"

        for label, obs_list in grouped_obs.items():
            for mu, cov, sigma_json in obs_list:
                blob = self._make_blob(mu, cov, label)
                det_msg.positions.append(Point(x=float(mu[0]), y=float(mu[1]), z=self.h_ground))
                det_msg.labels.append(label)
                det_msg.radii.append(float(sigma_json))
                det_msg.scale_x.append(blob.scale_x)
                det_msg.scale_y.append(blob.scale_y)
                det_msg.yaw.append(blob.yaw)

        self.detection_positions_pub.publish(det_msg)

        target_found = any(
            label_to_sigma.get(lbl, -1) == 0.0
            for lbl in grouped_obs.keys()
        )

        msg_target      = Bool()
        msg_target.data = target_found
        self.target_detected_pub.publish(msg_target)

        if target_found:
            self.get_logger().info("🧐 TARGET SPOTTED")

        return target_found


    def _update_field(self, pose, grouped_obs, all_projected_points):
        """Updates the belief field and publishes projected points."""
        footprint_mask = None
        fp = compute_ground_footprint(pose, self.camera, self.h_ground, self.max_projection_dist)
        if fp.is_valid and fp.area > 0.1:
            footprint_mask = rasterize_polygon(fp, self._field)

        with self._coverage_mask_lock:
            coverage_mask = (self._latest_coverage_mask.copy()
                            if self._latest_coverage_mask is not None else None)

        if all_projected_points:
            pts_concat       = np.vstack(all_projected_points)
            pts_xyz          = np.empty((pts_concat.shape[0], 3), dtype=np.float32)
            pts_xyz[:, :2]   = pts_concat
            pts_xyz[:, 2]    = self.h_ground
            self._publish_projected_points(pts_xyz)

        self._field.update_evidence(
            grouped_obs,
            footprint_mask=footprint_mask,
            coverage_mask=coverage_mask,
        )


    def _extract_roi(self, request):
        """Extracts ROI from request if valid."""
        if request.roi_x_max > request.roi_x_min and request.roi_y_max > request.roi_y_min:
            return (request.roi_x_min, request.roi_x_max, request.roi_y_min, request.roi_y_max)
        return None


    def _publish_all_viz(self, frame_bgr, vlm_detections, blobs_viz, roi):
        """Publishes all RViz visualizations."""
        with self._prompts_lock:
            label_to_sigma = dict(self._label_to_sigma)

        self._publish_annotated_bboxes(frame_bgr, vlm_detections, label_to_sigma)
        self._publish_field(roi=roi)
        self._publish_grid_for_rviz()
        self._publish_blob_markers(blobs_viz, label_to_sigma)
        self._publish_target_estimate()
    # -----------------------------------------------------------------------
    
    # -----------------------------------------------------------------------
    # PUBLISHERS
    # -----------------------------------------------------------------------
    def _publish_field(self, roi=None):
        """Publishes the posterior field as a PosteriorFieldMsg, optionally cropped to the ROI."""
        F = self._field.get_posterior_probability()
        nx_full, ny_full = self._field.nx, self._field.ny
        origin_x = self._field.x_min
        origin_y = self._field.y_min

        # If the ROI is valid, crop the field to the ROI bounds and adjust the origin accordingly
        if roi is not None:
            x_min, x_max, y_min, y_max = roi
            j0 = max(0, int((x_min - self._field.x_min) / self._field.delta))
            j1 = min(nx_full, int((x_max - self._field.x_min) / self._field.delta) + 1)
            i0 = max(0, int((y_min - self._field.y_min) / self._field.delta))
            i1 = min(ny_full, int((y_max - self._field.y_min) / self._field.delta) + 1)
            F = F[i0:i1, j0:j1]
            origin_x = self._field.x_min + j0 * self._field.delta
            origin_y = self._field.y_min + i0 * self._field.delta

        msg = PosteriorFieldMsg()
        msg.origin_x   = float(origin_x)
        msg.origin_y   = float(origin_y)
        msg.resolution = float(self._field.delta)
        msg.width      = int(F.shape[1])
        msg.height     = int(F.shape[0])
        msg.data       = F.flatten().astype(np.float32).tolist()
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
        """Publishes the posterior field as an OccupancyGrid for RViz visualization.
        F is already a probability in [0, 1] by construction, so we map it directly
        to the OccupancyGrid [0, 100] range without dynamic renormalisation.
        """
        F = self._field.get_posterior_probability()
        if F.size == 0:
            return
        data_int = np.clip(F * 100.0, 0, 100).astype(np.int8)

        og = OccupancyGrid()
        og.header.frame_id = "world"
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
        clear.header.frame_id = "world"
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        z = self.h_ground + 0.06
        for k, blob in enumerate(blobs):
            # Convert the yaw angle to a quaternion for rviz
            qw = float(np.cos(blob.yaw / 2.0))
            qz = float(np.sin(blob.yaw / 2.0))

            # Internal marker (internal ellipse)
            m_inner = Marker()
            m_inner.header.frame_id = "world"
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
                m_outer.header.frame_id = "world"
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

    # TODO: 
    def _publish_target_estimate(self):
        log_F = self._field.get_log_F()
        finite_mask = np.isfinite(log_F)
        if not finite_mask.any(): return

        log_F_max = log_F[finite_mask].max()
        if log_F_max < -10.0: return

        log_F_masked = np.where(finite_mask, log_F, -np.inf)
        idx = np.unravel_index(np.argmax(log_F_masked), log_F.shape)
        x_est, y_est = self._field.idx_to_world(idx[0], idx[1])

        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "target_estimate"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x_est), float(y_est), float(self.h_ground + 0.5)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 1.0
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 1.0
        self.target_estimate_pub.publish(m)

    def _publish_projected_points(self, pts_xyz: np.ndarray):
        """Publishes ground-projected sample points as PointCloud2 for RViz debugging."""
        if pts_xyz.size == 0:
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "world"

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