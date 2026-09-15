import argparse
import json
import os
import time

# Set Rerun memory limits to avoid dropping old log messages (such as the pinhole camera details)
os.environ["RERUN_SERVER_MEMORY_LIMIT"] = "4GB"

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

# Helpers
def quat_to_mat3(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)  ],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)  ],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])

def ros_image_to_numpy(msg) -> np.ndarray:
    h, w = msg.height, msg.width
    enc  = msg.encoding
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc in ("bgr8", "rgb8"):
        img = data.reshape(h, w, 3)
        if enc == "bgr8":
            img = img[:, :, ::-1].copy()
    elif enc == "mono8":
        img = data.reshape(h, w)
    else:
        img = data.reshape(h, w, -1)
    return img

def read_pointcloud2_xyz(msg):
    step = msg.point_step
    raw  = bytes(msg.data)
    if not raw or step < 12:
        return np.zeros((0, 3), dtype=np.float32)
    # Vectorized point cloud reading using numpy views
    pts = np.frombuffer(raw, dtype=np.uint8).reshape(-1, step)[:, :12].copy().view(np.float32)
    return pts

def occupancy_grid_to_pointcloud(msg, color_fn_vec, z_height=0.01):
    """
    Convert a nav_msgs/OccupancyGrid to a 3D point cloud (vectorized).
    color_fn_vec(values) -> numpy array of [r, g, b, a] colors.
    """
    w   = int(msg.info.width)
    h   = int(msg.info.height)
    ox  = float(msg.info.origin.position.x)
    oy  = float(msg.info.origin.position.y)
    res = float(msg.info.resolution)
    raw = np.array(msg.data, dtype=np.int8).reshape(h, w)

    y_idx, x_idx = np.where(raw > 0)
    if len(x_idx) == 0:
        return None, None, res
        
    pts = np.zeros((len(x_idx), 3), dtype=np.float32)
    pts[:, 0] = ox + (x_idx + 0.5) * res
    pts[:, 1] = oy + (y_idx + 0.5) * res
    pts[:, 2] = z_height

    vals = raw[y_idx, x_idx]
    clrs = color_fn_vec(vals)
    return pts, clrs, res

def marker_color(c, default_alpha=200):
    return [
        int(c.r * 255),
        int(c.g * 255),
        int(c.b * 255),
        int(c.a * 255) if c.a > 0 else default_alpha,
    ]

def marker_points_to_numpy(points):
    return np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)

def log_marker(entity_path: str, marker, flat_cylinder: bool = False):
    """Log a single visualization_msgs/Marker to Rerun."""
    ARROW       = 0
    CUBE        = 1
    SPHERE      = 2
    CYLINDER    = 3
    LINE_STRIP  = 4
    LINE_LIST   = 5
    SPHERE_LIST = 7
    POINTS      = 8
    TEXT        = 9

    mtype  = marker.type
    action = marker.action

    if action in (2, 3):
        rr.log(entity_path, rr.Clear(recursive=(action == 3)))
        return

    color = marker_color(marker.color)

    if mtype == CUBE:
        pos = [marker.pose.position.x,
               marker.pose.position.y,
               0.01 if flat_cylinder else marker.pose.position.z]
        q = marker.pose.orientation
        rr.log(entity_path,
               rr.Boxes3D(
                   centers=[pos],
                   half_sizes=[[marker.scale.x * 0.5,
                                marker.scale.y * 0.5,
                                0.02 if flat_cylinder else marker.scale.z * 0.5]],
                   colors=[color],
                   rotations=[[q.x, q.y, q.z, q.w]]
               ))

    elif mtype in (SPHERE, CYLINDER):
        pos = [marker.pose.position.x, marker.pose.position.y, 0.01 if flat_cylinder else marker.pose.position.z]
        if flat_cylinder:
            rr.log(entity_path, rr.Ellipsoids3D(centers=[pos], half_sizes=[[marker.scale.x * 0.5, marker.scale.y * 0.5, 0.02]], colors=[color],))
        else:
            rr.log(entity_path, rr.Points3D([pos], colors=[color], radii=marker.scale.x * 0.5))

    elif mtype in (SPHERE_LIST, POINTS):
        if marker.points:
            pts = marker_points_to_numpy(marker.points)
            colors = ([marker_color(c) for c in marker.colors] if marker.colors else [color] * len(pts))
            if flat_cylinder:
                pts = pts.copy()
                pts[:, 2] = 0.01
                half_sizes = np.column_stack([
                    np.full(len(pts), marker.scale.x * 0.5),
                    np.full(len(pts), marker.scale.y * 0.5),
                    np.full(len(pts), 0.02),
                ])
                rr.log(entity_path, rr.Ellipsoids3D(centers=pts, half_sizes=half_sizes, colors=colors))
            else:
                rr.log(entity_path, rr.Points3D(pts, colors=colors, radii=marker.scale.x * 0.5))

    elif mtype == ARROW:
        if marker.points and len(marker.points) >= 2:
            start = [marker.points[0].x, marker.points[0].y, marker.points[0].z]
            end   = [marker.points[1].x, marker.points[1].y, marker.points[1].z]
            rr.log(entity_path, rr.Arrows3D(origins=[start], vectors=[[end[0]-start[0], end[1]-start[1], end[2]-start[2]]], colors=[color], radii=marker.scale.y * 0.5))
        else:
            pos = [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z]
            q   = marker.pose.orientation
            R   = quat_to_mat3(q)
            vec = R @ np.array([marker.scale.x, 0, 0])
            rr.log(entity_path, rr.Arrows3D(origins=[pos], vectors=[vec.tolist()], colors=[color], radii=marker.scale.y * 0.5))

    elif mtype == LINE_STRIP:
        if marker.points:
            pts = marker_points_to_numpy(marker.points)
            rr.log(entity_path, rr.LineStrips3D([pts], colors=[color], radii=marker.scale.x * 0.5))

    elif mtype == LINE_LIST:
        if marker.points and len(marker.points) >= 2:
            pts    = marker_points_to_numpy(marker.points)
            strips = [pts[i:i+2] for i in range(0, len(pts) - 1, 2)]
            rr.log(entity_path, rr.LineStrips3D(strips, colors=[color] * len(strips), radii=marker.scale.x * 0.5))

    elif mtype == TEXT:
        pos = [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z]
        rr.log(entity_path, rr.Points3D([pos], colors=[color], radii=0.05, labels=[marker.text]))

def log_marker_array(base_path: str, msg, flat_cylinder: bool = False):
    for marker in msg.markers:
        path = (f"{base_path}/{marker.ns}/{marker.id}" if marker.ns else f"{base_path}/{marker.id}")
        log_marker(path, marker, flat_cylinder=flat_cylinder)


def register_custom_types(typestore):
    add = {}
    add.update(get_types_from_msg(
        "std_msgs/Header header\n"
        "geometry_msgs/Point[] positions\nstring[] labels\n"
        "float32[] radii\nfloat32[] scale_x\nfloat32[] scale_y\nfloat32[] yaw",
        "interfaces/msg/DetectionPositions",
    ))
    add.update(get_types_from_msg(
        "float32[] xs\nfloat32[] ys\nfloat64 area",
        "interfaces/msg/CoveragePolygon",
    ))
    add.update(get_types_from_msg(
        "std_msgs/Header header\n"
        "float64 origin_x\nfloat64 origin_y\nfloat64 resolution\n"
        "uint32 width\nuint32 height\nfloat32[] grid\n"
        "interfaces/CoveragePolygon[] polygons\nfloat64 total_area\nuint32 num_polygons",
        "interfaces/msg/CoverageField",
    ))
    typestore.register(add)

# --- Visualizer Class ---
class RerunBagVisualizer:
    def __init__(self, typestore, step_by_step=False, cam_subsample=5):
        self.typestore = typestore
        self.step_by_step = step_by_step
        self.cam_subsample = cam_subsample
        self.camera_frame_idx = 0
        
        # State
        self.pinhole_logged = False
        self.acq_count = 0
        self.ann_count = 0
        self.post_count = 0
        self.det_count = 0
        self.target_detected = False
        self.waypoint_positions = []
        self.last_drone_z = None
        
        # Dynamic TF Tree
        self.tf_parents = {}      # child_frame_id -> parent_frame_id
        self.tf_transforms = {}   # child_frame_id -> (translation, rotation_matrix)

        # Topic -> Handler mapping
        self.topic_handlers = {
            "/tf": self.handle_tf,
            "/tf_static": self.handle_tf,
            "/camera/camera/color/camera_info": self.handle_camera_info,
            "/camera/camera/color/image_raw/compressed": self.handle_camera_compressed,
            "/vrpn_mocap/jetson_nx/pose": self.handle_vrpn_pose,
            "/acquired_img": self.handle_acquired_img,
            "/annotated_img": self.handle_annotated_img,
            "/posterior_grid": self.handle_posterior_grid,
            "/coverage_grid": self.handle_coverage_grid,
            "/detection_positions": self.handle_detection_positions,
            "/projected_bbox_points": self.handle_projected_bbox_points,
            "/semantic_blobs": self.handle_semantic_blobs,
            "/exploration_active_domain": self.handle_active_domain,
            "/exploration_candidates": self.handle_candidates,
            "/exploration_chosen_waypoint": self.handle_chosen_waypoint,
            "/target_estimate_marker": self.handle_target_estimate,
            "/target_detected": self.handle_target_detected,
            "/ground_footprints": self.handle_ground_footprints,
            "/current_footprint": self.handle_current_footprint,
        }

    # ── TF Methods ──
    def update_tf(self, msg):
        for t in msg.transforms:
            child = t.child_frame_id
            parent = t.header.frame_id
            tr    = t.transform.translation
            q     = t.transform.rotation
            
            self.tf_parents[child] = parent
            self.tf_transforms[child] = (np.array([tr.x, tr.y, tr.z]), quat_to_mat3(q))
            
            if child in ("jetson_nx", "base_link"):
                self.last_drone_z = tr.z

    def get_world_pose(self, frame_id):
        """Recursively computes the world pose of a frame by traversing up the TF tree."""
        curr_frame = frame_id
        pos = np.zeros(3)
        R = np.eye(3)
        visited = set()
        
        while curr_frame in self.tf_transforms:
            if curr_frame in visited:
                return None, None
            visited.add(curr_frame)
            
            t_pos, t_R = self.tf_transforms[curr_frame]
            pos = t_pos + t_R @ pos
            R = t_R @ R
            
            parent = self.tf_parents.get(curr_frame)
            if parent in ("world", "map"):
                return pos, R
            if parent is None:
                return None, None
            curr_frame = parent
            
        return None, None

    def update_camera_pose(self):
        pos_cam, R_cam = self.get_world_pose("camera_color_optical_frame")
        if pos_cam is not None:
            rr.log("drone/camera", rr.Transform3D(translation=pos_cam.tolist(), mat3x3=R_cam))

    # --- Handlers ---
    def handle_tf(self, msg):
        self.update_tf(msg)
        self.update_camera_pose()

    def handle_camera_info(self, msg):
        if not self.pinhole_logged:
            fx, fy = float(msg.k[0]), float(msg.k[4])
            cx, cy = float(msg.k[2]), float(msg.k[5])
            iw, ih = int(msg.width), int(msg.height)
            rr.log("drone/camera",
                   rr.Pinhole(focal_length=[fx, fy],
                              principal_point=[cx, cy],
                              width=iw, height=ih,
                              image_plane_distance=0.5),
                   static=True)
            self.pinhole_logged = True
            print(f"[camera_info] Pinhole: {fx:.1f}/{fy:.1f} {cx:.1f}/{cy:.1f} {iw}x{ih}")

    def handle_camera_compressed(self, msg):
        self.camera_frame_idx += 1
        if self.cam_subsample > 1 and self.camera_frame_idx % self.cam_subsample != 0:
            return
        try:
            img_component = rr.EncodedImage(contents=bytes(msg.data), media_type="image/jpeg")
        except AttributeError:
            img_component = rr.ImageEncoded(contents=bytes(msg.data), media_type="image/jpeg")
        rr.log("drone/camera", img_component)

    def handle_vrpn_pose(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation
        pos = np.array([p.x, p.y, p.z])
        R = quat_to_mat3(q)
        self.last_drone_z = p.z
        
        self.tf_parents["jetson_nx"] = "world"
        self.tf_transforms["jetson_nx"] = (pos, R)
        
        rr.log("drone/body", rr.Transform3D(translation=pos.tolist(), mat3x3=R))
        self.update_camera_pose()

    def handle_acquired_img(self, msg):
        img = ros_image_to_numpy(msg)
        self.acq_count += 1
        rr.log("ui/acquired", rr.Image(img))
        
        # Accumulate the actual camera pose at this step as a waypoint
        pos_cam, _ = self.get_world_pose("camera_color_optical_frame")
        if pos_cam is not None:
            wp_pos = pos_cam.tolist()
            if not self.waypoint_positions or not np.allclose(self.waypoint_positions[-1], wp_pos, atol=1e-3):
                self.waypoint_positions.append(wp_pos)
                rr.log("mission/trajectory", rr.LineStrips3D([self.waypoint_positions], colors=[[30, 100, 200]]))
                labels = [f"w{i+1}" for i in range(len(self.waypoint_positions))]
                rr.log("mission/waypoints", rr.Points3D(self.waypoint_positions,
                                                    colors=[[30, 100, 200]] * len(self.waypoint_positions),
                                                    radii=0.06,
                                                    labels=labels))
        
        if self.step_by_step:
            time.sleep(2.0)

    def handle_annotated_img(self, msg):
        img = ros_image_to_numpy(msg)
        self.ann_count += 1
        rr.log("ui/annotated", rr.Image(img))

    def handle_posterior_grid(self, msg):
        def posterior_color_vec(vals):
            intensity = vals / 100.0
            # Warm color map (Yellow -> Orange -> Red) for high-contrast visibility against coverage
            r = np.full_like(vals, 255, dtype=np.uint8)
            g = (np.clip((1.0 - intensity) * 200 + 30, 0, 255)).astype(np.uint8)
            b = (np.clip((1.0 - intensity) * 100, 0, 255)).astype(np.uint8)
            a = (np.clip((0.3 + 0.7 * intensity) * 220, 0, 255)).astype(np.uint8)
            return np.column_stack([r, g, b, a])
            
        # Set z_height=0.015 to prevent z-fighting and overlay above the coverage grid
        pts, clrs, res = occupancy_grid_to_pointcloud(msg, posterior_color_vec, z_height=0.015)
        if pts is not None:
            rr.log("map/posterior_grid", rr.Points3D(pts, colors=clrs, radii=res * 0.4))
        self.post_count += 1

    def handle_coverage_grid(self, msg):
        def coverage_color_vec(vals):
            # Cool mint/teal color for coverage grid
            clrs = np.zeros((len(vals), 4), dtype=np.uint8)
            clrs[:, 0] = 30   # R
            clrs[:, 1] = 180  # G
            clrs[:, 2] = 140  # B
            clrs[:, 3] = 90   # A
            return clrs
            
        # Set z_height=0.005 to prevent z-fighting and lay it on the floor
        pts, clrs, res = occupancy_grid_to_pointcloud(msg, coverage_color_vec, z_height=0.005)
        if pts is not None:
            rr.log("map/coverage_grid", rr.Points3D(pts, colors=clrs, radii=res * 0.45))

    def handle_detection_positions(self, msg):
        pts = [[p.x, p.y, 0.05] for p in msg.positions]
        lbls = list(msg.labels)
        if pts:
            rr.log("map/detections", rr.Points3D(np.array(pts, dtype=np.float32), colors=[[255, 80, 0]] * len(pts), radii=0.12, labels=lbls))
        self.det_count += 1

    def handle_projected_bbox_points(self, msg):
        pts = read_pointcloud2_xyz(msg)
        if len(pts):
            rr.log("map/projected_points", rr.Points3D(pts, colors=[[200, 200, 255, 180]] * len(pts), radii=0.03))

    def handle_semantic_blobs(self, msg):
        log_marker_array("map/semantic_blobs", msg, flat_cylinder=True)

    def handle_active_domain(self, msg):
        log_marker("planner/active_domain", msg)

    def handle_candidates(self, msg):
        log_marker_array("planner/candidates", msg)

    def handle_chosen_waypoint(self, msg):
        log_marker("planner/chosen_waypoint", msg)

    def handle_target_detected(self, msg):
        self.target_detected = msg.data

    def handle_target_estimate(self, msg):
        if self.target_detected:
            log_marker("map/target_estimate", msg, flat_cylinder=True)
        else:
            rr.log("map/target_estimate", rr.Clear(recursive=True))

    def handle_ground_footprints(self, msg):
        log_marker_array("map/ground_footprints", msg)

    def handle_current_footprint(self, msg):
        log_marker("map/current_footprint", msg)

    #  Runner
    def run(self, bag_path):
        print(f"Opening bag: {bag_path}")

        with Reader(bag_path) as reader:
            connections = [c for c in reader.connections if c.topic in self.topic_handlers]

            for connection, timestamp_ns, rawdata in reader.messages(connections=connections):
                rr.set_time("ros_time", timestamp=timestamp_ns * 1e-9)

                topic = connection.topic
                try:
                    msg = self.typestore.deserialize_cdr(rawdata, connection.msgtype)
                    self.topic_handlers[topic](msg)
                except Exception as e:
                    print(f"[{topic}] Error: {e}")

        print(f"\nDone.")
        print(f"  Pinhole logged:   {self.pinhole_logged}")
        print(f"  Acquired images:  {self.acq_count}")
        print(f"  Annotated images: {self.ann_count}")
        print(f"  Posterior fields: {self.post_count}")
        print(f"  Detection msgs:   {self.det_count}")

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag",            required=True)
    parser.add_argument("--json",           default=None, help="Optional path to the mission JSON report")
    parser.add_argument("--step-by-step",   action="store_true")
    parser.add_argument("--cam-subsample",  type=int, default=5, help="Subsample rate for camera frames (log 1 every N frames)")
    args = parser.parse_args()

    instruction = None
    result = None

    if args.json:
        with open(args.json) as f:
            report = json.load(f)

        feedback    = report["runs"][0]["feedback"]
        result      = report["runs"][0]["result"]
        instruction = report["mission"]["instruction"]


    # Blueprint: Creates a 3D view of the world, and two 2D views of the annotated and acquired images
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name="3D Scene",
                origin="/",
                contents=["/**", "-ui/**"]
            ),
            rrb.Vertical(
                rrb.Spatial2DView(name="Annotated", origin="ui/annotated"),
                rrb.Spatial2DView(name="Acquired",  origin="ui/acquired"),
            ),
            column_shares=[3, 1],
        ),
        collapse_panels=True,
    )

    rr.init("semantic_exploration", spawn=True)
    rr.send_blueprint(blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


    if args.json:
        for fb in reversed(feedback):
            target_est = fb.get("target_estimate")
            if target_est is not None:
                tx = target_est.get("x")
                ty = target_est.get("y")
                if tx is not None and ty is not None:
                    rr.log("mission/target_estimate_json", rr.Points3D([[tx, ty, 0.0]], colors=[[255, 140, 0]], radii=0.15), static=True)
                    break


    # Typestore: Used to deserialize ROS 2 messages 
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    register_custom_types(typestore)

    # Visualization: Runs the visualizer 
    visualizer = RerunBagVisualizer(typestore, step_by_step=args.step_by_step, cam_subsample=args.cam_subsample)
    visualizer.run(args.bag)

    if instruction and result:
        print(f"\nInstruction: '{instruction}'")
        print(f"Result: target_found={result['target_found']}, steps={result['steps']}, dist={result['distance_travelled']:.2f}m")

if __name__ == "__main__":
    main()