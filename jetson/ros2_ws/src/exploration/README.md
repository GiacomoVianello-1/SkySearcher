# Exploration Package

ROS 2 exploration tools for AirSim. This package contains geometric utilities, a ground coverage mapper, a frontier-based planner, and an optional visual-language-model (VLM) node used to extract semantic clues from images.

## Available Nodes / Console Scripts

- **coverage_map_node**: computes the ground footprint from camera frustums, accumulates a cumulative coverage polygon, publishes visualization markers, and exposes the `/acquire_observation` service (std_srvs/Trigger). See [exploration/coverage_map_node.py](exploration/coverage_map_node.py#L1).
- **planner_node**: the exploration planner that samples frontier candidates, evaluates information gain (geometric + semantic), and sends movement commands. It exposes the `/start_mission` action (`interfaces/action/StartMission.action`) plus the `/stop_mission` service, and uses several service clients (`/acquire_observation`, `/update_posterior`, move/takeoff services, `/configure_prompts`). See [exploration/planner.py](exploration/planner.py#L1).
- **mission_manager_node**: action client for `/start_mission` that can optionally spawn the drone, logs feedback, and writes a mission report JSON. See [exploration/mission_manager.py](exploration/mission_manager.py#L1).
- **vlm_node**: a Qwen-based VLM service that answers `detect_semantic_clues` requests by returning bounding boxes for requested categories. The node publishes `vlm_busy` while processing and exposes the `detect_semantic_clues` service. See [exploration/vlm_node.py](exploration/vlm_node.py#L1).
- **panoramic_scout_node**: panoramic scout helper used during mission setup. See [exploration/panoramic_scout.py](exploration/panoramic_scout.py#L1).
- **semantic_belief_tracker_node**: maintains the posterior field and publishes semantic belief updates. See [exploration/semantic_belief_tracker.py](exploration/semantic_belief_tracker.py#L1).
- **test_vlm_client**: small client utility for testing the VLM service. See [exploration/test_vlm_client.py](exploration/test_vlm_client.py#L1).
- **camera / utils**: shared modules used by the planner and coverage map logic. See [exploration/camera.py](exploration/camera.py#L1) and [exploration/utils.py](exploration/utils.py#L1).

Notes:
- The repository also contains `camera_frustum_node.py` which visualizes frustums and exact intersection volumes, but it is not exported as a console script in `setup.py`. You can run it directly from source (e.g. `python3 -m exploration.camera_frustum_node` with the package on `PYTHONPATH`) or add an entry-point in `setup.py` to make it available via `ros2 run`.

## Launch files

- `launch/exploration.launch.py` — launches the coverage map and planner (and a semantic map node if present). See [launch/exploration.launch.py](launch/exploration.launch.py#L1).
- `launch/exploration_mission.launch.py` — mission launch that includes the VLM, coverage map, semantic belief tracker and planner. The target instruction is now passed when sending the `/start_mission` action goal. See [launch/exploration_mission.launch.py](launch/exploration_mission.launch.py#L1).

Run the mission launch (example):

```bash
# source your ROS 2 workspace after building
ros2 launch exploration exploration_mission.launch.py

# In another terminal, send the action goal with the target instruction
ros2 action send_goal /start_mission interfaces/action/StartMission "{instruction: 'red rowing boat in the lake'}"
```

Or run individual nodes from the installed package:

```bash
ros2 run exploration coverage_map_node
ros2 run exploration planner_node
ros2 run exploration vlm_node
```

If you prefer to run a script directly from source (for development) you can run it as a module:

```bash
python3 -m exploration.camera_frustum_node
```

## Key topics, services, and parameters

- Topics published (examples): `/camera_frustum_markers`, `/frustum_intersections`, `/ground_footprints`, `/current_footprint`, `/exploration_candidates`, `/exploration_chosen_waypoint`, `/exploration_active_domain`, `vlm_busy`.
- Topics subscribed (examples): camera info and odometry topics configured via parameters (defaults: `/airsim_node/Drone1/bottom_center_Scene/camera_info`, `/airsim_node/Drone1/odom_local`), and `/posterior_field`, `/target_detected`.
- Services provided: `/acquire_observation` (coverage_map_node, std_srvs/Trigger), `/stop_mission` (planner_node), `detect_semantic_clues` (vlm_node, interfaces/DetectSemantics).
- Actions provided: `/start_mission` (planner_node, `StartMission`).
- Planner service clients: `/acquire_observation`, `/update_posterior`, `/configure_prompts`, `/airsim_node/Drone1/move_to_position`, `/airsim_node/Drone1/takeoff`.

Important parameters (examples):

- `camera_frame`, `camera_info_topic`, `odom_topic` — frame/topic names used for TF and camera info.
- `h_ground`, `depth`, `max_projection_dist` — used by `coverage_map_node` to project frustums onto the ground plane.
- Planner parameters: `instruction`, `vlm_log_path`, `flight_altitude`, `max_projection_dist`, `active_domain_radius`, `pi_0`, `gamma`, `frontier_spacing`, `move_velocity`, `move_timeout_sec`, `max_steps`, `target_confirmation_threshold`, `debug_planner`.
- VLM parameter: `max_bbox_area_ratio` — discard overly-large predicted boxes.

See the top of each node file for the full list of parameters and defaults (e.g. [exploration/coverage_map_node.py](exploration/coverage_map_node.py#L1), [exploration/planner.py](exploration/planner.py#L1), [exploration/vlm_node.py](exploration/vlm_node.py#L1)).

## Dependencies

The package uses the following (non-exhaustive) dependencies:

- `rclpy`, ROS 2 message/service packages
- `numpy`, `scipy`, `shapely`, `matplotlib`
- `cosysairsim` (AirSim client wrappers used by utilities)
- VLM stack: `torch`, `transformers`, `qwen_vl_utils`, `opencv-python`, `Pillow`, `cv_bridge` (for `vlm_node`)
- Local message/service packages: `interfaces`, `airsim_interfaces`

Install system and Python dependencies in your workspace environment before running the nodes.

## Overview of implementation

- `camera.py`: camera intrinsics, frustum rays, plane representation, frustum intersection helpers and plotting utilities.
- `utils.py`: TF helpers, AirSim pose conversion, camera creation from AirSim intrinsics, and `compute_ground_footprint()` used by the map and planner.
- `coverage_map_node.py`: computes per-observation footprint, publishes current footprint and cumulative boundary, and serves `/acquire_observation`.
- `planner.py`: frontier sampling, information-gain evaluation (π_0 geometric baseline + semantic posterior integration), candidate scoring, movement command logic, and mission control loop.
- `mission_manager.py`: client-side mission orchestration, optional spawn, feedback logging, and final JSON report export.
- `panoramic_scout.py`: panoramic scout support for VLM-based heading selection.
- `semantic_belief_tracker.py`: posterior field maintenance and update service.
- `test_vlm_client.py`: quick VLM service test client.
- `vlm_node.py`: wraps a Qwen VLM to extract bounding boxes for requested target categories and serves `detect_semantic_clues` requests.
- `camera.py` and `utils.py`: shared geometry and TF helpers.

