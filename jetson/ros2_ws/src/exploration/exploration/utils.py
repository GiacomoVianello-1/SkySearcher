import numpy as np
from shapely.geometry import Polygon, MultiPoint
from tf2_ros import Buffer
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import rclpy

def quaternion_to_R(q):
    """Convert quaternion to rotation matrix."""
    w, x, y, z = q.w, q.x, q.y, q.z
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])

# --- GEOMETRY UTILS ---

def lookup_camera_pose(tf_buffer: Buffer, world_frame: str, camera_frame: str):
    """
    Lookup the current camera pose in the world frame from the TF tree.

    Returns
    -------
    T_world_cam : np.ndarray (4, 4) or None
        Homogeneous transform world <-- camera, or None if the lookup fails.
    yaw : float or None
        Yaw of the *body* frame in the world frame (rad), extracted from the
        odom_local --> world transform. None on failure.

    Notes
    -----
    Two separate TF lookups are needed:
      - world <-- camera_optical  (for footprint projection)
      - world <-- odom_local      (for yaw, which belongs to the body frame)
    Both are done here so callers can decide which they need.
    """
    try:
        tf_cam = tf_buffer.lookup_transform(world_frame, camera_frame, rclpy.time.Time())
        q_cam = tf_cam.transform.rotation
        t_cam = tf_cam.transform.translation
        T = np.eye(4)
        T[:3, :3] = quaternion_to_R(q_cam)
        T[:3, 3] = [t_cam.x, t_cam.y, t_cam.z]
        return T
    except (LookupException, ConnectivityException, ExtrapolationException):
        return None
    

def lookup_body_yaw(tf_buffer: Buffer, world_frame: str, body_frame: str) -> float | None:
    """
    Extract the body yaw (rotation about world Z) from TF.
    Returns the yaw in radians, or None if the lookup fails.
    """
    try:
        tf_body = tf_buffer.lookup_transform(world_frame, body_frame, rclpy.time.Time())
        R_body = quaternion_to_R(tf_body.transform.rotation)
        return float(np.arctan2(R_body[1, 0], R_body[0, 0]))
    except (LookupException, ConnectivityException, ExtrapolationException):
        return None
    

def lookup_static_transform(tf_buffer: Buffer, parent: str, child: str):
    """
    Lookup a static (or one-shot) transform between two frames. Returns a 4x4 np.ndarray or None.
    Identical to lookup_camera_pose but with generic frame names, useful for the body --> optical static offset.
    """
    try:
        tf = tf_buffer.lookup_transform(parent, child, rclpy.time.Time())
        q = tf.transform.rotation
        t = tf.transform.translation
        T = np.eye(4)
        T[:3, :3] = quaternion_to_R(q)
        T[:3, 3] = [t.x, t.y, t.z]
        return T
    except (LookupException, ConnectivityException, ExtrapolationException):
        return None
    

def build_candidate_pose(x: float, y: float, yaw: float, altitude: float, T_body_to_opt: np.ndarray) -> np.ndarray:
    """
    Build the 4x4 camera optical pose for a *candidate* waypoint (x, y, yaw).

    The drone body is placed at (x, y, altitude) with the given yaw; the
    optical frame is derived by composing with the static body-->optical offset.

    Parameters
    ----------
    x, y          : candidate XY position in world frame
    yaw           : desired body yaw (rad) in world frame
    altitude      : flight altitude (Z in world frame)
    T_body_to_opt : static 4x4 transform body <-- optical (from TF)

    Returns
    -------
    T_world_to_opt: np.ndarray (4, 4)
    """
    T_world_to_body = np.eye(4)
    T_world_to_body[:3, :3] = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw),  np.cos(yaw), 0.0],
        [0.0,          0.0,         1.0],
    ])
    T_world_to_body[:3, 3] = [x, y, altitude]
    return T_world_to_body @ T_body_to_opt


def compute_ground_footprint(pose: np.ndarray, camera, h_ground: float = 0.0, max_projection_dist: float = 30.0) -> Polygon:
    """
    Compute the 2D ground footprint of the camera frustum by intersecting the 4 corner rays with the ground plane z = h_ground.

    Handles rays parallel to the ground or pointing upward by clamping to max_projection_dist along the XY direction of the ray.

    INPUT:
    pose               : 4x4 world <-- camera homogeneous transform
    camera             : Camera instance (uses camera.fov_rays and camera.depth)
    h_ground           : Z coordinate of the ground plane in world frame
    max_projection_dist: fallback clamp distance (m) when the ray does not reach the ground within a finite distance

    OUTPUT:
    footprint : shapely.geometry.Polygon  (may be invalid/empty on degenerate input)
    """
    R = pose[:3, :3]
    t = pose[:3, 3]

    corners_world = (camera.fov_rays * camera.depth) @ R.T + t
    tl3, tr3, bl3, br3 = corners_world

    def _intersect(corner3):
        ray = corner3 - t
        dz = ray[2]
        ray_xy = ray[:2]
        norm_xy = np.linalg.norm(ray_xy)
        if norm_xy < 1e-6:
            return (float(t[0]), float(t[1]))
        dir_xy = ray_xy / norm_xy
        if abs(dz) < 1e-6 or (h_ground - t[2]) / dz <= 0:
            pt_xy = t[:2] + dir_xy * max_projection_dist
            return (float(pt_xy[0]), float(pt_xy[1]))
        s = (h_ground - t[2]) / dz
        pt = t + s * ray
        if np.linalg.norm(pt[:2] - t[:2]) > max_projection_dist:
            pt_xy = t[:2] + dir_xy * max_projection_dist
            return (float(pt_xy[0]), float(pt_xy[1]))
        return (float(pt[0]), float(pt[1]))

    tl = _intersect(tl3)
    tr = _intersect(tr3)
    bl = _intersect(bl3)
    br = _intersect(br3)
    return MultiPoint([tl, tr, bl, br]).convex_hull