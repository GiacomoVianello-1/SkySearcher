import numpy as np
from shapely.geometry import Polygon, MultiPoint
from tf2_ros import Buffer
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException, TransformException
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

def lookup_transform_at(tf_buffer: Buffer, parent: str, child: str, stamp, timeout_sec: float = 0.1):
    """
    Lookup parent <-- child at the sensor timestamp `stamp` (builtin_interfaces/Time).

    Waits up to timeout_sec for the transform to arrive. Returns a 4x4 np.ndarray, or
    None if it is unavailable at that instant (never falls back to the latest pose).
    """
    try:
        tf = tf_buffer.lookup_transform(
            parent, child,
            rclpy.time.Time.from_msg(stamp),
            timeout=rclpy.duration.Duration(seconds=float(timeout_sec)),
        )
    except TransformException:  # base class of Lookup/Connectivity/Extrapolation
        return None
    q = tf.transform.rotation
    t = tf.transform.translation
    T = np.eye(4)
    T[:3, :3] = quaternion_to_R(q)
    T[:3, 3] = [t.x, t.y, t.z]
    return T


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
    Lookup the latest transform between two frames. Returns a 4x4 np.ndarray or None.
    For time-invariant offsets (body --> optical) and for "where is the drone now" queries.
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
    Return T_world_opt for a candidate body pose.

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
    T_world_to_opt : 4x4 pose of the optical frame expressed in the world frame.
    """
    c, s = np.cos(yaw), np.sin(yaw)

    T_world_to_body = np.eye(4)
    T_world_to_body[:3, :3] = np.array([
        [ c,   -s,  0.0],
        [ s,    c,  0.0],
        [0.0, 0.0,  1.0],
    ])
    T_world_to_body[:3, 3] = [x, y, altitude]
    return T_world_to_body @ T_body_to_opt

def compute_ground_footprint(pose: np.ndarray, camera, h_ground: float = 0.0, max_projection_dist: float = 30.0) -> Polygon:
    """
    Compute the 2D ground footprint of the camera frustum by intersecting the 4 corner rays with the ground plane z = h_ground.

    Handles rays parallel to the ground or pointing upward by clamping to max_projection_dist along the XY direction of the ray.

    Parameters
    ----------
    pose               : 4x4 world <-- camera homogeneous transform
    camera             : Camera instance (uses camera.fov_rays and camera.depth)
    h_ground           : Z coordinate of the ground plane in world frame
    max_projection_dist: fallback clamp distance (m) when the ray does not reach the ground within a finite distance

    Returns
    -------
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


def project_bbox_montecarlo(xmin: float, ymin: float, xmax: float, ymax: float,
                            pose: np.ndarray, camera, h_ground: float = 0.0,
                            n_samples: int = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Projects the 2D bounding box onto the ground plane using Monte Carlo ray-casting.
    Samples points within the bbox, casts rays from the camera, and finds their intersection with the ground plane z = h_ground.
    Returns the geometric mean and covariance of the intersection points in world coordinates.
    """
    rng = np.random.default_rng()
    u = rng.uniform(xmin, xmax, size=n_samples)
    v = rng.uniform(ymin, ymax, size=n_samples)

    # Pinhole camera model: convert pixel coordinates to normalized camera coordinates
    x_opt = (u - camera.cx) / camera.fx
    y_opt = (v - camera.cy) / camera.fy
    z_opt = np.ones_like(x_opt)
    
    rays_opt = np.stack([x_opt, y_opt, z_opt], axis=0)
    R, t = pose[:3, :3], pose[:3, 3]
    rays_world = R.dot(rays_opt)

    ray_z = rays_world[2, :]
    lambda_intersect = (h_ground - t[2]) / ray_z

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

    # Additive Isotropic Floor (regularization + minimum uncertainty of the detector)
    sigma_min = 0.5  # meters — minimum uncertainty due to pixel quantization + bbox jitter
    cov = np.cov(pts_final) + (sigma_min**2) * np.eye(2)

    # Upper Cap ONLY as safety net (proportional, preserves the shape)
    eigvals, eigvecs = np.linalg.eigh(cov)
    sigma_max = 50.0
    if eigvals.max() > sigma_max**2:
        eigvals = eigvals * (sigma_max**2 / eigvals.max())
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return mean_xy, cov, pts_final.T