from __future__ import annotations
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point as ShapelyPoint
from geometry_msgs.msg import Pose2D

from .utils import compute_ground_footprint


def generate_boundary_candidates(
    coverage: Polygon | MultiPolygon | None,
    map_polygon: Polygon,
    spacing: float,
    current_x: float,
    current_y: float,
    camera=None,
    pose=None,
    h_ground: float = 0.0,
    max_projection_dist: float = 30.0,
    include_inward: bool = True,
) -> list[tuple[float, float, float]]:
    """
    Sample equispaced points along the exterior boundary of the covered area.
    Computes strict OUTWARD normals, and optionally INWARD (anti-normal) directions st include_inward=True.
    Fixes boundary wrapping tangent bugs via symmetric finite differences.
    """
    out = []

    # Handle bootstrap / initial state
    if coverage is None or coverage.is_empty:
        if camera is not None and pose is not None:
            current_fp = compute_ground_footprint(pose, camera, h_ground, max_projection_dist)
            if current_fp.is_valid and current_fp.area > 0.1:
                target_poly = current_fp
            else:
                target_poly = ShapelyPoint(current_x, current_y).buffer(15.0)
        else:
            target_poly = ShapelyPoint(current_x, current_y).buffer(15.0)
        polys = [target_poly]
    else:
        polys = list(coverage.geoms) if isinstance(coverage, MultiPolygon) else [coverage]

    for poly in polys:
        if poly.is_empty:
            continue
        bnd = poly.exterior
        L = bnd.length
        if L <= 0:
            continue

        n = max(4, int(np.ceil(L / spacing)))
        for i in range(n):
            s = (i / n) * L
            pt = bnd.interpolate(s)

            # Ensure candidate is within map bounds
            if not (map_polygon.contains(pt) or map_polygon.intersects(pt)):
                continue

            # Robust tangent computation via symmetric finite differences
            ds = 0.1  # small offset in meters
            s_next = min(L, s + ds)
            s_prev = max(0.0, s - ds)
            if abs(s_next - s_prev) < 1e-6:
                continue

            p_next = bnd.interpolate(s_next)
            p_prev = bnd.interpolate(s_prev)

            tx = p_next.x - p_prev.x
            ty = p_next.y - p_prev.y
            tnorm = np.hypot(tx, ty) + 1e-9
            nx, ny = ty / tnorm, -tx / tnorm

            # Test probe to ensure normal points OUTWARD
            test_pt = ShapelyPoint(pt.x + 0.5 * nx, pt.y + 0.5 * ny)
            if poly.contains(test_pt):
                nx, ny = -nx, -ny

            yaw_outward = float(np.arctan2(ny, nx))
            yaw_inward = float(np.arctan2(-ny, -nx))
            yaw_tangent1 = float(np.arctan2(ty, tx))
            yaw_tangent2 = float(np.arctan2(-ty, -tx))

            for yaw_cand in (yaw_outward, yaw_tangent1, yaw_tangent2, yaw_inward if include_inward else None):
                if yaw_cand is not None:
                    out.append((float(pt.x), float(pt.y), yaw_cand))


    return out


def generate_semantic_candidates(
    detection_mus: list[np.ndarray],
    standoff_dist: float,
    map_polygon: Polygon,
    min_cluster_dist: float = 3.0,
) -> list[tuple[float, float, float]]:
    """
    Generate candidate viewpoints facing detected objects (mu_d), after spatial clustering
    to remove redundant/duplicate detection points.
    """
    if not detection_mus:
        return []

    # 1. Spatial clustering of detection centers
    clusters: list[np.ndarray] = []
    for mu in detection_mus:
        if len(mu) < 2:
            continue
        mu_2d = np.array([float(mu[0]), float(mu[1])])
        
        is_merged = False
        for c in clusters:
            if np.hypot(mu_2d[0] - c[0], mu_2d[1] - c[1]) < min_cluster_dist:
                is_merged = True
                break
        if not is_merged:
            clusters.append(mu_2d)

    # 2. Generate 4 orthogonal viewpoints facing each cluster center
    candidates = []
    n_angles = 4
    for c in clusters:
        target_x, target_y = c[0], c[1]
        for i in range(n_angles):
            yaw = float(i * (2.0 * np.pi) / n_angles)
            cand_x = target_x - standoff_dist * np.cos(yaw)
            cand_y = target_y - standoff_dist * np.sin(yaw)

            cand_pt = ShapelyPoint(cand_x, cand_y)
            if map_polygon.contains(cand_pt) or map_polygon.intersects(cand_pt):
                candidates.append((cand_x, cand_y, yaw))

    return candidates


def generate_all_candidates(
    coverage: Polygon | MultiPolygon | None,
    map_polygon: Polygon,
    detection_mus: list[np.ndarray],
    current_x: float,
    current_y: float,
    camera=None,
    pose=None,
    h_ground: float = 0.0,
    max_projection_dist: float = 30.0,
    frontier_spacing: float = 20.0,
    standoff_dist: float = 8.0,
    include_inward: bool = True,
) -> list[Pose2D]:
    """
    Combines boundary and semantic candidates, returning a list of geometry_msgs.msg.Pose2D objects.
    """
    boundary_cands = generate_boundary_candidates(
        coverage=coverage,
        map_polygon=map_polygon,
        spacing=frontier_spacing,
        current_x=current_x,
        current_y=current_y,
        camera=camera,
        pose=pose,
        h_ground=h_ground,
        max_projection_dist=max_projection_dist,
        include_inward=include_inward,
    )

    # Generate semantic candidates facing detected objects from different perspective angles
    semantic_cands = generate_semantic_candidates(
        detection_mus=detection_mus,
        standoff_dist=standoff_dist,
        map_polygon=map_polygon,
    )

    rotation_cands = []
    n_angles = 4
    for i in range(n_angles):
        yaw = float(i * (2.0 * np.pi) / n_angles)
        rotation_cands.append((current_x, current_y, yaw))

    all_cands = boundary_cands + semantic_cands + rotation_cands
    pose_list = []
    for x, y, yaw in all_cands:
        p = Pose2D()
        p.x = float(x)
        p.y = float(y)
        p.theta = float(yaw)
        pose_list.append(p)

    return pose_list
