std_msgs/Header header

float32 origin_x       # world-frame X of the grid lower-left corner
float32 origin_y       # world-frame Y of the grid lower-left corner
float32 resolution     # cell size delta in metres

uint32 width           # number of columns (x direction)
uint32 height          # number of rows (y direction)

float32[] data         # row-major, length = width * height