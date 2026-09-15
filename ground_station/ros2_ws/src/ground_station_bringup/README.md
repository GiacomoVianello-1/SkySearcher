# Ground Station Bringup Package

TODO

## Use this Package

Inside the ground station container, first build the workspace:
```bash
cd ros2_ws
colcon build --packages-select ground_station_bringup
source install/setup.bash
```
Then launch the package via
```bash
ros2 launch ground_station_bringup ground_station.launch.py
```
This starts RViz only, which is what the outdoor RTK setup needs. For the indoor
arena, add `use_mocap:=true` to also start the OptiTrack VRPN client and its TF
broadcaster:
```bash
ros2 launch ground_station_bringup ground_station.launch.py use_mocap:=true
```
Adjust the parameters in the `ground_station.launch.py` as needed (you need to match the OptiTrack server IP).
The TF broadcaster parameters live in `config/tf_broadcaster.yaml`. The camera
extrinsic is published by the Jetson, from `camera_extrinsics_publisher` in
`jetson/ros2_ws/src/exploration_v2/config/exploration_params.yaml`; it must not
be published here as well, or `camera_link` ends up with two parents.

The RViz `Fixed Frame` is `odom`, matching `world_frame` in the Jetson config.
Set both to `world` for the indoor mocap arena.

## OptiTrack 

TODO

## TF Broadcaster

TODO