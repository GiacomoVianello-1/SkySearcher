import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import prepare, contains_xy

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from std_srvs.srv import Trigger

from exploration.camera import Camera
from exploration.utils import quaternion_to_R, compute_ground_footprint

from interfaces.srv import AcquireObservation
from interfaces.msg import CoverageField, CoveragePolygon

from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped


class CoverageMapNode(Node):
    def __init__(self):
        super().__init__("coverage_map_node")

        # --- PARAMETERS ---
        self.declare_parameter("world_frame",       "world")
        self.declare_parameter("camera_frame",      "camera_color_optical_frame")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("odom_topic",        "/vrpn_mocap/jetson_nx/pose")

        self.declare_parameter("h_ground", 0.0) # the ground plane altitude in the world frame     
        self.declare_parameter("max_projection_dist", 200.0)

        # Grid parameters
        self.declare_parameter("grid_x_min", -10.0)
        self.declare_parameter("grid_x_max",  10.0)
        self.declare_parameter("grid_y_min", -10.0)
        self.declare_parameter("grid_y_max",  10.0)
        self.declare_parameter("delta_grid",  0.1)

        self.world_frame  = self.get_parameter("world_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        odom_topic        = self.get_parameter("odom_topic").value
        self.h_ground             = float(self.get_parameter("h_ground").value)
        self.max_projection_dist  = float(self.get_parameter("max_projection_dist").value)

        x_min      = float(self.get_parameter("grid_x_min").value)
        x_max      = float(self.get_parameter("grid_x_max").value)
        y_min      = float(self.get_parameter("grid_y_min").value)
        y_max      = float(self.get_parameter("grid_y_max").value)
        delta_grid = float(self.get_parameter("delta_grid").value)

        # --- GRID STATE ---
        self._grid_x_min  = x_min
        self._grid_y_min  = y_min
        self._grid_delta  = delta_grid
        self._grid_nx     = int(np.ceil((x_max - x_min) / delta_grid))
        self._grid_ny     = int(np.ceil((y_max - y_min) / delta_grid))

        # Pre-compute cell-centre coordinate grids (reused at every acquire)
        _xs = x_min + (np.arange(self._grid_nx) + 0.5) * delta_grid
        _ys = y_min + (np.arange(self._grid_ny) + 0.5) * delta_grid
        self._XX, self._YY = np.meshgrid(_xs, _ys)   # shape (ny, nx)

        # --- COVERAGE STATE ---
        self.camera = None
        self.current_pose = None
        self.last_observation_pose = None
        self.coverage: Polygon | MultiPolygon | None = None

        # --- TF ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        mocap_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        
        # --- SUBSCRIBERS ---
        self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(PoseStamped, odom_topic, self.odom_callback, mocap_qos)

        # --- PUBLISHERS ---
        self.coverage_field_pub    = self.create_publisher(CoverageField, "/coverage_field",      1)
        self.footprint_pub         = self.create_publisher(MarkerArray,   "/ground_footprints",  10)
        self.current_footprint_pub = self.create_publisher(Marker,        "/current_footprint",  10)
        self.coverage_grid_pub     = self.create_publisher(OccupancyGrid, "/coverage_grid",       1)
        self.live_frustum_pub      = self.create_publisher(Marker,        "/live_frustum",       10)
        self._startup_clear_timer  = self.create_timer(1.0, self._startup_clear)

        # --- SERVICES ---
        self.acquire_srv = self.create_service(AcquireObservation, "/acquire_observation", self.acquire_observation_callback)
        self.reset_srv   = self.create_service(Trigger,            "/reset_coverage",      self.reset_coverage_callback)

        self.get_logger().info("Coverage Map Node started")

    # --- Callbacks ---

    def camera_info_callback(self, msg: CameraInfo):
        """Extract camera intrinsics from the first received CameraInfo message."""
        if self.camera is None:
            self.camera = Camera(
                h=msg.height, w=msg.width,
                fx=msg.k[0], fy=msg.k[4],
                cx=msg.k[2], cy=msg.k[5],
                depth=self.max_projection_dist,
            )
            self.get_logger().info("Camera intrinsics loaded")

    def odom_callback(self, msg: PoseStamped):
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

        # Publish the current camera frustum in RViz just for visualization
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
        response.success       = True
        response.message       = f"Area={self.coverage.area:.1f} m², Components={n_polys}"
        response.total_area    = float(self.coverage.area)
        response.num_polygons  = n_polys
        response.polygons      = polygons
        response.grid          = grid
        response.grid_width    = int(self._grid_nx)
        response.grid_height   = int(self._grid_ny)
        response.grid_origin_x = float(self._grid_x_min)
        response.grid_origin_y = float(self._grid_y_min)
        response.grid_resolution = float(self._grid_delta)

        # Publish topics
        self._publish_coverage_field_from_data(polygons, grid)
        self._publish_coverage_boundary()
        self._publish_current_footprint(footprint)

        return response

    def reset_coverage_callback(self, request, response):
        """Resets the cumulative coverage to empty and clears all RViz markers."""
        self.coverage = None
        self.last_observation_pose = None

        ma = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.world_frame
        clear.header.stamp    = self.get_clock().now().to_msg()
        clear.ns              = "cumulative_coverage"
        clear.action          = Marker.DELETEALL
        ma.markers.append(clear)
        self.footprint_pub.publish(ma)

        # Clear /current_footprint marker in RViz
        del_m = Marker()
        del_m.header.frame_id = self.world_frame
        del_m.header.stamp    = self.get_clock().now().to_msg()
        del_m.ns              = "current_footprint"
        del_m.id              = 0
        del_m.action          = Marker.DELETE
        self.current_footprint_pub.publish(del_m)

        self.get_logger().info("[reset_coverage] Coverage cleared.")
        response.success = True
        response.message = "Coverage reset."
        return response

    # --- Rasterisation and visualisation ---

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
            mask = contains_xy(part, pts_x, pts_y).reshape(
                self._grid_ny, self._grid_nx)
            grid[mask] = 1.0

        return grid.flatten().tolist()

    # --- RViz ---

    def _publish_coverage_field_from_data(self, polygons, grid):
        """Publish CoverageField topic using already calculated data."""
        msg = CoverageField()
        msg.header.frame_id = "world"
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
        self._publish_coverage_grid(
            grid_array=np.array(grid).reshape(self._grid_ny, self._grid_nx))

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

        z = self.h_ground + 0.05

        for idx, poly in enumerate(parts):
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns              = "cumulative_coverage"
            m.id              = idx
            m.type            = Marker.LINE_STRIP
            m.action          = Marker.ADD
            m.scale.x         = 0.05
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.5, 1.0

            coords = list(poly.exterior.coords)
            for x, y in coords:
                m.points.append(Point(x=float(x), y=float(y), z=z))
            if coords:
                m.points.append(Point(x=float(coords[0][0]), y=float(coords[0][1]), z=z))

            ma.markers.append(m)

        self.footprint_pub.publish(ma)

    def _publish_current_footprint(self, footprint: Polygon):
        del_m = Marker()
        del_m.header.frame_id = self.world_frame
        del_m.header.stamp    = self.get_clock().now().to_msg()
        del_m.ns              = "current_footprint"
        del_m.id              = 0
        del_m.action          = Marker.DELETE
        self.current_footprint_pub.publish(del_m)
        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns     = "current_footprint"
        m.id     = 0
        m.type   = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.05
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 1.0

        z = self.h_ground + 0.1 # avoid mesh clipping
        coords = list(footprint.exterior.coords)
        for x, y in coords:
            m.points.append(Point(x=float(x), y=float(y), z=z))
        if coords:
            m.points.append(Point(x=float(coords[0][0]), y=float(coords[0][1]), z=z))

        self.current_footprint_pub.publish(m)

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
        m.scale.x = 0.05
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 0.7 

        z = self.h_ground + 0.05
        coords = list(footprint.exterior.coords)
        for x, y in coords:
            m.points.append(Point(x=float(x), y=float(y), z=z))
        if coords:
            m.points.append(Point(x=float(coords[0][0]), y=float(coords[0][1]), z=z))

        self.live_frustum_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = CoverageMapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()