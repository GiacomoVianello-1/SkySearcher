# CoverageField.msg
# Published by coverage_map_node on /coverage_field after every acquire_observation.
# Carries both a rasterised binary grid (for numerical consumers) and the exact
# Shapely polygon geometry (for geometric consumers).
#
# Coordinate convention: ENU world frame, same as the planner.

std_msgs/Header header

# --- Raster grid (binary, float32 0.0/1.0) ---
# Laid out in row-major order: data[i * width + j] is the cell at row i, col j.
# Cell (i,j) centre is at (origin_x + (j+0.5)*resolution,
#                           origin_y + (i+0.5)*resolution).
float64 origin_x
float64 origin_y
float64 resolution
uint32  width
uint32  height
float32[] grid          # 0.0 = not covered, 1.0 = covered

# --- Polygon geometry (exact Shapely exterior rings) ---
# Each CoveragePolygon is one part of a Polygon or MultiPolygon.
# For a simple Polygon there is exactly one entry.
# For a MultiPolygon there are N entries (one per geom).
interfaces/CoveragePolygon[] polygons

# --- Summary ---
float64 total_area      # sum of all polygon areas (m²)
uint32  num_polygons    # len(polygons)