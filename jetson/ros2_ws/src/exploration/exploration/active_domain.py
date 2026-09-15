from __future__ import annotations
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point as ShapelyPoint
from shapely.geometry.base import BaseGeometry
from shapely import prepare, contains_xy

from typing import Protocol

class GridLike(Protocol):
    """Minimal interface: anything with a regular 2D grid of cell centres."""
    nx: int
    ny: int
    XX: np.ndarray  # shape (ny, nx)
    YY: np.ndarray  # shape (ny, nx)


def compute_active_domain(coverage: BaseGeometry | None,
                        map_polygon: BaseGeometry,
                          R_t: float,
                          bootstrap_centre: tuple[float, float] | None = None,
                          bootstrap_radius: float = 20.0) -> BaseGeometry:
    """
    Build the active domain from the cumulative coverage.

    Behaviour
    ---------
    - If coverage is None or empty: return a disc around `bootstrap_centre`
      (typically the drone's initial position) of radius `bootstrap_radius`.
      This gives the planner something to chew on at step 0.
    - Else: D_t = buffer(C_t, R_t) \\ C_t — a ring of thickness R_t around
      the coverage perimeter.

    Returns a Shapely geometry (Polygon or MultiPolygon). Empty geometries
    are returned as-is; the caller should check `.is_empty`.
    """
    if R_t <= 0:
        raise ValueError(f"R_t must be > 0 (got {R_t})")

    if coverage is None or coverage.is_empty:
        if bootstrap_centre is None:
            return Polygon()  # empty
        cx, cy = bootstrap_centre
        return ShapelyPoint(cx, cy).buffer(bootstrap_radius)

    extended = coverage.buffer(R_t)
    domain = extended.difference(coverage)
    domain = domain.intersection(map_polygon)
    return domain


def rasterize_polygon(poly: BaseGeometry, field: GridLike) -> np.ndarray:
    """
    Convert a Shapely (Multi)Polygon to a boolean mask aligned with `field`'s
    grid. A cell (i, j) is True iff its centre lies inside `poly`.
    """
    mask = np.zeros((field.ny, field.nx), dtype=bool)
    if poly is None or poly.is_empty:
        return mask

    pts_x = field.XX.ravel()
    pts_y = field.YY.ravel()

    if isinstance(poly, MultiPolygon):
        parts = list(poly.geoms)
    elif isinstance(poly, Polygon):
        parts = [poly]
    else:
        parts = [g for g in getattr(poly, "geoms", [poly]) if isinstance(g, Polygon)]

    for part in parts:
        if part.is_empty:
            continue
        prepare(part)
        part_mask = contains_xy(part, pts_x, pts_y).reshape(field.ny, field.nx)
        mask |= part_mask

    return mask