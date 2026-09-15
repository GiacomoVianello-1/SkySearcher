# Visualizer

This directory contains the visualization suite used for analyzing both simulated and real-world robotic exploration experiments. We leverage [Rerun](https://rerun.io/) for live 3D visualization of ROS 2 bags, providing a highly flexible and performant alternative to RViz.

## 📁 Included Scripts

The directory includes two main scripts for analysis and visualization:

- `plot_mission_results.py`: Generates PDF figures and plots from the JSON mission logs.
- `rerun_visualizer.py`: Reads a recorded ROS 2 bag and dynamically replays the 3D exploration mission, recreating the drone's trajectory, camera feeds, and semantic maps.

## ⚙️ Requirements

Before running the visualizer, ensure you have the required Python dependencies installed:

```bash
pip install rerun-sdk rosbags numpy
```

*(Note: If you are running this inside the `ground_station` Docker container, these dependencies are already installed).*

## 🎒 ROS Bags Data

To visualize a mission, you need the corresponding recorded ROS 2 bags. 
Download the pre-recorded `ros_bags` from [here](TODO) and place them inside a `ros_bags` directory as follows:

```text
ros_bags/
 ├── bag_1/
 ├── bag_2/
 └── ...
```

<details>
<summary><b>How to record a custom ROS Bag</b></summary>

If you want to record a ROS bag for a new run, we suggest recording the following topics to ensure full compatibility with the visualizer:

```bash
ros2 bag record \
  /acquired_img /annotated_img \
  /camera/camera/color/camera_info \
  /camera/camera/color/image_raw/compressed \
  /coverage_field /coverage_grid /coverage_grid_updates \
  /current_footprint /current_footprint_array \
  /detection_positions /exploration_active_domain \
  /exploration_active_domain_array /exploration_candidates \
  /exploration_chosen_waypoint /exploration_chosen_waypoint_array \
  /goal_pose /ground_footprints /initialpose \
  /live_frustum /live_frustum_array /parameter_events \
  /posterior_field /posterior_grid /posterior_grid_updates \
  /projected_bbox_points /semantic_blobs /target_detected \
  /target_estimate_marker /tf /tf_static \
  /vrpn_mocap/jetson_nx/pose \
  -o /root/ros2_ws/mission_reports/bag_1
``` 
Avoid recording the uncompressed camera stream, as the ROS bag will grow too quickly.

</details>

## 👁️ Running the Rerun Visualizer

The `rerun_visualizer.py` script reads the temporal data directly from the `.db3` bag files and correlates it with the static mission data in the JSON report.

Inside your environment (or the `ground_station` Docker container), run:

```bash
python3 rerun_visualizer.py \
    --bag ros_bags/bag_1 \
    --json json_files/[GOOD]real_mission_report_20260612_165057.json
```

### Additional Options
- `--step-by-step`: Adds a small delay during image acquisitions, making it easier to follow the drone's decision-making process in real-time.

```bash
python3 rerun_visualizer.py --bag <path> --json <path> --step-by-step
```