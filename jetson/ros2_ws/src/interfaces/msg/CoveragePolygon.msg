# CoveragePolygon.msg
# One polygon part (exterior ring only — holes are not tracked).
# Coordinates are in the ENU world frame.
# The ring is NOT closed: last point != first point (Shapely convention).

float64[] xs   # x coordinates of exterior ring vertices
float64[] ys   # y coordinates of exterior ring vertices
float64   area # pre-computed area of this polygon part (m²)