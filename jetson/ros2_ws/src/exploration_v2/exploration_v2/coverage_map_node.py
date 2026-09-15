import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import prepare, contains_xy

from .camera import Camera
from .utils import quaternion_to_R, compute_ground_footprint

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from nav_msgs.msg import OccupancyGrid
from std_srvs.srv import Trigger

from interfaces.srv import AcquireObservation, GenerateWaypoints, ResetBelief
from interfaces.msg import CoverageField, CoveragePolygon, DetectionPositions
from .waypoint_generator import generate_all_candidates
from .active_domain import compute_active_domain
import threading


class CoverageMapNode(Node):
    def __init__(self):
        super().__init__("coverage_map_node")

        self.cb_group = ReentrantCallbackGroup()
        self._last_frustum_pub_time = 0.0

        # --- PARAMETERS ---
        self.declare_parameter("camera_frame",      "Drone1/bottom_center_optical")
        self.declare_parameter("camera_info_topic", "/airsim_node/Drone1/bottom_center_Scene/camera_info")
        self.declare_parameter("odom_topic",        "/airsim_node/Drone1/odom_local")
        self.declare_parameter("world_frame",       "world")
        self.declare_parameter("h_ground", 0.0)
        self.declare_parameter("max_projection_dist", 200.0)

        # Grid parameters
        self.declare_parameter("grid_x_min", -350.0)
        self.declare_parameter("grid_x_max",  672.0)
        self.declare_parameter("grid_y_min", -377.0)
        self.declare_parameter("grid_y_max",  450.0)
        self.declare_parameter("delta_grid",   10.0)
        self.declare_parameter("active_domain_radius", 24.0)
        self.declare_parameter("marker_scale", 1.0)

        self.camera_frame = self.get_parameter("camera_frame").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        odom_topic        = self.get_parameter("odom_topic").value
        self.world_frame  = self.get_parameter("world_frame").value
        self.h_ground             = float(self.get_parameter("h_ground").value)
        self.max_projection_dist  = float(self.get_parameter("max_projection_dist").value)
        self.active_domain_radius = float(self.get_parameter("active_domain_radius").value)
        self.marker_scale = float(self.get_parameter("marker_scale").value)

        x_min      = float(self.get_parameter("grid_x_min").value)
        x_max      = float(self.get_parameter("grid_x_max").value)
        y_min      = float(self.get_parameter("grid_y_min").value)
        y_max      = float(self.get_parameter("grid_y_max").value)
        delta_grid = float(self.get_parameter("delta_grid").value)

        # --- GRID STATE ---
        self._grid_x_min  = x_min
        self._grid_x_max  = x_max
        self._grid_y_min  = y_min
        self._grid_y_max  = y_max
        self._grid_delta  = delta_grid
        self._grid_nx     = int(np.ceil((x_max - x_min) / delta_grid))
        self._grid_ny     = int(np.ceil((y_max - y_min) / delta_grid))

        # Pre-compute cell-centre coordinate grids (reused at every acquire)
        _xs = x_min + (np.arange(self._grid_nx) + 0.5) * delta_grid
        _ys = y_min + (np.arange(self._grid_ny) + 0.5) * delta_grid
        self._XX, self._YY = np.meshgrid(_xs, _ys)   # shape (ny, nx)

        self.map_polygon = Polygon([
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
        ])

        # --- COVERAGE STATE ---
        self.camera = None
        self.current_pose = None
        self.last_observation_pose = None
        self.coverage: Polygon | MultiPolygon | None = None
        self._detection_mus: list[np.ndarray] = []
        self._posterior_peak: np.ndarray | None = None
        self._detection_lock = threading.Lock()

        # --- TF ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- SUBSCRIBERS ---
        self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(Odometry,   odom_topic,        self.odom_callback,        QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(DetectionPositions, "/detection_positions", self._detection_positions_cb, 1)
        self.create_subscription(Marker, "/target_estimate_marker", self._target_estimate_cb, 1)

        # --- PUBLISHERS ---
        self.coverage_field_pub    = self.create_publisher(CoverageField,   "/coverage_field",     1)
        self.live_frustum_pub      = self.create_publisher(Marker,          "/live_frustum",      10)
        self.footprint_pub         = self.create_publisher(MarkerArray,     "/ground_footprints", 10)
        self.coverage_grid_pub     = self.create_publisher(OccupancyGrid,   "/coverage_grid",      1)
        self.candidates_pub        = self.create_publisher(MarkerArray,     "/exploration_candidates", 10)
        self.active_domain_pub     = self.create_publisher(Marker,          "/exploration_active_domain", 10)
        self._startup_clear_timer  = self.create_timer(1.0, self._startup_clear)

        # --- SERVICES ---
        self.acquire_srv = self.create_service(AcquireObservation, "/acquire_observation", self.acquire_observation_callback, callback_group=self.cb_group)
        self.reset_srv = self.create_service(Trigger, "/reset_coverage", self.reset_coverage_callback, callback_group=self.cb_group)
        self.reset_belief_srv = self.create_service(ResetBelief, "/reset_coverage_belief", self.reset_coverage_belief_callback, callback_group=self.cb_group)
        self.generate_wp_srv = self.create_service(GenerateWaypoints, "/generate_waypoints", self.generate_waypoints_callback, callback_group=self.cb_group)

        self.get_logger().info("Coverage Map Node started")

    def _detection_positions_cb(self, msg: DetectionPositions):
        mus = [np.array([p.x, p.y]) for p in msg.positions]
        with self._detection_lock:
            self._detection_mus = mus

    def generate_waypoints_callback(self, request: GenerateWaypoints.Request, response: GenerateWaypoints.Response) -> GenerateWaypoints.Response:
        frontier_spacing = request.frontier_spacing if request.frontier_spacing > 0.0 else 20.0
        standoff_dist = request.standoff_dist if request.standoff_dist > 0.0 else 8.0

        with self._detection_lock:
            mus = list(self._detection_mus)

        # Inject the peak history into detection mus
        if self._posterior_peak is not None:
            mus.append(self._posterior_peak)

        candidates = generate_all_candidates(
            coverage=self.coverage,
            map_polygon=self.map_polygon,
            detection_mus=mus,
            current_x=request.current_x,
            current_y=request.current_y,
            camera=self.camera,
            pose=self.current_pose,
            h_ground=self.h_ground,
            max_projection_dist=self.max_projection_dist,
            frontier_spacing=frontier_spacing,
            standoff_dist=standoff_dist,
        )

        self._publish_candidate_markers(candidates)

        response.success = True
        response.message = f"Generated {len(candidates)} candidate waypoints"
        response.candidates = candidates
        return response

    # --- Callbacks ---

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
        try:
            tf = self.tf_buffer.lookup_transform(self.world_frame, self.camera_frame, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return

        tr = tf.transform.translation
        q  = tf.transform.rotation
        t  = np.array([tr.x, tr.y, tr.z])
        R  = quaternion_to_R(q)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = t
        self.current_pose = T

        # Publish the current camera frustum in RViz at max 10Hz to avoid CPU starvation
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if (now_sec - self._last_frustum_pub_time) >= 0.1:
            self._last_frustum_pub_time = now_sec
            if self.camera is not None:
                self._publish_live_frustum(self.current_pose)

    def acquire_observation_callback(self, request, response):
        if self.camera is None or self.current_pose is None:
            response.success = False
            response.message = "Camera or pose not yet available"
            return response

        pose = self.current_pose.copy()

        footprint = compute_ground_footprint(pose, self.camera, self.h_ground, self.max_projection_dist)

        if not footprint.is_valid or footprint.area <= 0.1:
            response.success = False
            response.message = f"Invalid footprint (area={footprint.area:.2f})"
            return response

        # Fuse into cumulative coverage
        if self.coverage is None:
            self.coverage = footprint
        else:
            eps = self._grid_delta * 0.5 # small buffer to ensure proper merging of adjacent polygons
            merged = unary_union([self.coverage.buffer(eps), footprint.buffer(eps)])
            self.coverage = merged.buffer(-eps).simplify(0.1, preserve_topology=True)

        self.last_observation_pose = pose

        n_polys = (len(self.coverage.geoms)
                if isinstance(self.coverage, MultiPolygon) else 1)

        # Rasterise
        grid = self._rasterise_coverage()

        # Serialise polygons
        parts = (list(self.coverage.geoms)
                if isinstance(self.coverage, MultiPolygon)
                else [self.coverage])

        polygons = []
        for part in parts:
            coords = list(part.exterior.coords)
            if coords and coords[0] == coords[-1]:
                coords = coords[:-1]
            cp = CoveragePolygon()
            cp.xs   = [float(x) for x, _ in coords]
            cp.ys   = [float(y) for _, y in coords]
            cp.area = float(part.area)
            polygons.append(cp)

        # Fill response
        response.success            = True
        response.message            = f"Area={self.coverage.area:.1f} m², Components={n_polys}"
        response.total_area         = float(self.coverage.area)
        response.num_polygons       = n_polys
        response.polygons           = polygons
        response.grid_width         = int(self._grid_nx)
        response.grid_height        = int(self._grid_ny)
        response.grid_origin_x      = float(self._grid_x_min)
        response.grid_origin_y      = float(self._grid_y_min)
        response.grid_resolution    = float(self._grid_delta)

        # Publish topics and RViz markers
        self._publish_coverage_field_from_data(polygons, grid)
        self._publish_coverage_boundary()
        self._publish_active_domain()

        return response

    def reset_coverage_callback(self, request, response):
        """Resets the cumulative coverage to empty and clears all RViz markers."""
        self.coverage = None
        self.last_observation_pose = None

        # Clear /ground_footprints markers in RViz.
        ma = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.world_frame
        clear.header.stamp    = self.get_clock().now().to_msg()
        clear.ns              = "cumulative_coverage"
        clear.action          = Marker.DELETEALL
        ma.markers.append(clear)
        self.footprint_pub.publish(ma)

        self.get_logger().info("[reset_coverage] Coverage cleared.")
        response.success = True
        response.message = "Coverage reset."
        return response

    def reset_coverage_belief_callback(self, request, response):
        """Resets the cumulative coverage and updates grid bounds if provided."""
        self.coverage = None
        self.last_observation_pose = None
        with self._detection_lock:
            self._detection_mus = []

        if request.grid_x_max > request.grid_x_min:
            x_min = float(request.grid_x_min)
            x_max = float(request.grid_x_max)
            y_min = float(request.grid_y_min)
            y_max = float(request.grid_y_max)
            delta_grid = self._grid_delta

            self._grid_x_min = x_min
            self._grid_x_max = x_max
            self._grid_y_min = y_min
            self._grid_y_max = y_max
            self._grid_nx = int(np.ceil((x_max - x_min) / delta_grid))
            self._grid_ny = int(np.ceil((y_max - y_min) / delta_grid))

            _xs = x_min + (np.arange(self._grid_nx) + 0.5) * delta_grid
            _ys = y_min + (np.arange(self._grid_ny) + 0.5) * delta_grid
            self._XX, self._YY = np.meshgrid(_xs, _ys)

            self.map_polygon = Polygon([
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max),
            ])

        ma = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.world_frame
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.ns = "cumulative_coverage"
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        self.footprint_pub.publish(ma)

        self.get_logger().info("[reset_coverage_belief] Coverage & grid bounds reset.")
        response.success = True
        response.message = "Coverage reset."
        return response
    
    def _target_estimate_cb(self, msg: Marker):
        """Save the peak of the posterior map."""
        if msg.action == Marker.DELETE or msg.action == Marker.DELETEALL:
            self._posterior_peak = None
        else:
            self._posterior_peak = np.array([msg.pose.position.x, msg.pose.position.y])

    # --- Rasterisation and Visualisation ---

    def _rasterise_coverage(self) -> list:
        """
        Rasterises self.coverage onto the node's grid.
        Returns a flat float32 list of length ny*nx (row-major).
        Uses shapely.contains_xy for vectorised point-in-polygon.
        """

        grid = np.zeros((self._grid_ny, self._grid_nx), dtype=np.float32)

        if self.coverage is None or self.coverage.is_empty:
            return grid.flatten().tolist()

        parts = (list(self.coverage.geoms)
                 if isinstance(self.coverage, MultiPolygon)
                 else [self.coverage])

        pts_x = self._XX.ravel()
        pts_y = self._YY.ravel()

        for part in parts:
            if part.is_empty:
                continue
            prepare(part)
            mask = contains_xy(part, pts_x, pts_y).reshape(self._grid_ny, self._grid_nx)
            grid[mask] = 1.0

        return grid.flatten().tolist()

    # --- RViz ---

    def _publish_coverage_field_from_data(self, polygons, grid):
        """Publish CoverageField topic using already calculated data."""
        msg = CoverageField()
        msg.header.frame_id = self.world_frame
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.origin_x        = float(self._grid_x_min)
        msg.origin_y        = float(self._grid_y_min)
        msg.resolution      = float(self._grid_delta)
        msg.width           = int(self._grid_nx)
        msg.height          = int(self._grid_ny)
        msg.grid            = grid
        msg.polygons        = polygons
        msg.total_area      = float(self.coverage.area)
        msg.num_polygons    = len(polygons)
        self.coverage_field_pub.publish(msg)
        self._publish_coverage_grid(grid_array=np.array(grid).reshape(self._grid_ny, self._grid_nx))

    def _publish_coverage_boundary(self):
        if self.coverage is None:
            return

        ma = MarkerArray()

        # Clear all previous markers
        clear = Marker()
        clear.header.frame_id = self.world_frame
        clear.header.stamp    = self.get_clock().now().to_msg()
        clear.ns              = "cumulative_coverage"
        clear.id              = -1
        clear.action          = Marker.DELETEALL
        ma.markers.append(clear)

        parts = (list(self.coverage.geoms)
                if isinstance(self.coverage, MultiPolygon)
                else [self.coverage])

        z = self.h_ground + 0.05 # slightly above ground to avoid z-fighting with the mesh

        for idx, poly in enumerate(parts):
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns              = "cumulative_coverage"
            m.id              = idx
            m.type            = Marker.LINE_STRIP
            m.action          = Marker.ADD
            m.scale.x         = 1.0 * self.marker_scale
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.5, 1.0

            coords = list(poly.exterior.coords)
            for x, y in coords:
                m.points.append(Point(x=float(x), y=float(y), z=z))
            if coords:
                m.points.append(Point(x=float(coords[0][0]), y=float(coords[0][1]), z=z))

            ma.markers.append(m)

        self.footprint_pub.publish(ma)

    def _publish_coverage_grid(self, grid_array: np.ndarray):
        og = OccupancyGrid()
        og.header.frame_id = self.world_frame
        og.header.stamp    = self.get_clock().now().to_msg()
        og.info.resolution = float(self._grid_delta)
        og.info.width      = int(self._grid_nx)
        og.info.height     = int(self._grid_ny)
        og.info.origin.position.x = float(self._grid_x_min)
        og.info.origin.position.y = float(self._grid_y_min)
        og.info.origin.position.z = float(self.h_ground + 0.01)
        og.info.origin.orientation.w = 1.0
        og.data = (grid_array * 100).astype(np.int8).flatten().tolist()
        self.coverage_grid_pub.publish(og)

    def _startup_clear(self):
        self._startup_clear_timer.cancel()
        ma = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.world_frame
        clear.header.stamp    = self.get_clock().now().to_msg()
        clear.ns              = "cumulative_coverage"
        clear.id              = -1
        clear.action          = Marker.DELETEALL
        ma.markers.append(clear)
        self.footprint_pub.publish(ma)
    
    def _publish_live_frustum(self, pose):
        """Publish the current camera frustum as a polygon in RViz for visualization. Useful for manual control."""
        if pose is None or (pose[2, 3] - self.h_ground) < 0.5:
            return

        footprint = compute_ground_footprint(pose, self.camera, self.h_ground, self.max_projection_dist)
        if not footprint.is_valid or footprint.is_empty:
            return

        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns     = "live_frustum"
        m.id     = 0
        m.type   = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.5 * self.marker_scale
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 1.0 

        z = self.h_ground + 0.1
        coords = list(footprint.exterior.coords)
        for x, y in coords:
            m.points.append(Point(x=float(x), y=float(y), z=z))
        if coords:
            m.points.append(Point(x=float(coords[0][0]), y=float(coords[0][1]), z=z))
        self.live_frustum_pub.publish(m)
    
    def _publish_candidate_markers(self, candidates):
        ma = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        ma.markers.append(clear_marker)

        now = self.get_clock().now().to_msg()
        for i, cand in enumerate(candidates):
            x, y, yaw = float(cand.x), float(cand.y), float(cand.theta)

            # Sphere
            m_sphere = Marker()
            m_sphere.header.frame_id = self.world_frame
            m_sphere.header.stamp = now
            m_sphere.ns = "pts"
            m_sphere.id = i
            m_sphere.type = Marker.SPHERE
            m_sphere.pose.position.x = x
            m_sphere.pose.position.y = y
            m_sphere.pose.position.z = self.h_ground + 0.2
            m_sphere.scale.x = m_sphere.scale.y = m_sphere.scale.z = 0.8 * self.marker_scale
            m_sphere.color.r, m_sphere.color.g, m_sphere.color.b, m_sphere.color.a = 0.8, 0.8, 0.8, 1.0
            ma.markers.append(m_sphere)

            # Arrow
            m_arrow = Marker()
            m_arrow.header.frame_id = self.world_frame
            m_arrow.header.stamp = now
            m_arrow.ns = "yaw_arrows"
            m_arrow.id = i + 1000
            m_arrow.type = Marker.ARROW
            m_arrow.pose.position.x = x
            m_arrow.pose.position.y = y
            m_arrow.pose.position.z = self.h_ground + 0.3
            m_arrow.pose.orientation.z = float(np.sin(yaw / 2.0))
            m_arrow.pose.orientation.w = float(np.cos(yaw / 2.0))
            m_arrow.scale.x, m_arrow.scale.y, m_arrow.scale.z = 2.0 * self.marker_scale, 0.4 * self.marker_scale, 0.4 * self.marker_scale
            m_arrow.color.r, m_arrow.color.g, m_arrow.color.b, m_arrow.color.a = 1.0, 1.0, 1.0, 1.0
            ma.markers.append(m_arrow)

        self.candidates_pub.publish(ma)

    def _publish_active_domain(self):
        if self.current_pose is not None:
            cx, cy = float(self.current_pose[0, 3]), float(self.current_pose[1, 3])
        else:
            cx, cy = 0.0, 0.0

        domain = compute_active_domain(self.coverage, self.map_polygon,
                                       R_t=self.active_domain_radius,
                                       bootstrap_centre=(cx, cy),
                                       bootstrap_radius=self.active_domain_radius)
        if domain is None or domain.is_empty:
            return

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

        polys = domain.geoms if isinstance(domain, MultiPolygon) else [domain]
        for p in polys:
            coords = list(p.exterior.coords)
            for i in range(len(coords) - 1):
                p1, p2 = coords[i], coords[i+1]
                m.points.append(Point(x=float(p1[0]), y=float(p1[1]), z=z))
                m.points.append(Point(x=float(p2[0]), y=float(p2[1]), z=z))
        self.active_domain_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = CoverageMapNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()