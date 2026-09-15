# 🛸 exploration_v2 — ROS 2 Package

The **`exploration_v2`** package implements an autonomous exploration and semantic target localization system for Unmanned Aerial Vehicles (UAVs / drones). It combines a Vision-Language Model (VLM Qwen-VL), a 2D spatial grid Bayes belief filter, an Information Gain (IG) based planner, and a mission manager (`mission_manager_node`) designed for real-world flight testing.

## 🛠️ Environment Setup & Building

Before running any node or calling services, source your ROS 2 environment and workspace:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
```

To build the package with symlink installation:
```bash
python3 -m colcon build --symlink-install --packages-select exploration_v2
```

To launch the exploration nodes:
```bash
ros2 launch exploration_v2 exploration_mission.launch.py
```

### Running under tmuxinator

The container starts idle. Exec in, then start the stack:

```bash
cd jetson
docker compose up -d
docker compose exec robotics_ia bash
cd /root/ros2_ws
tmuxinator start -p tmuxinator/launch_mission.yaml
```

Layouts live in `jetson/ros2_ws/tmuxinator/`, so they come in with the existing `./ros2_ws` mount. Windows: `drivers` (mavros, camera, camera_tf), `perception` (vlm, coverage, belief), `mission` (planner, servoing), `bag` (`bagrec [name]`), `control` (service commands, echoed). Ctrl-C a pane to restart one node.

The panes use `ros2 run`, so each needs `--params-file` and `-r __node:=<name>`. The remap is not optional: `vlm_node`, `semantic_belief_tracker_node` and `planner_node` register as `qwen_vlm_node`, `semantic_map_node` and `exploration_planner_node`, and a node whose name misses its YAML section gets only `/**`, with no error.

Ground station: same layout, `ground_station/ros2_ws/tmuxinator/launch_ground_station.yaml`.

## 🧭 Frames

```
map ──(MAVROS local_position)──> base_link ──(static)──> camera_link ──> camera_color_optical_frame
```

`map` is ENU per REP-105: x East, y North, z Up. `tf.send` defaults to false and
resets on every MAVROS start, so set it each time or the topic publishes while
TF stays empty and every arrival wait returns `NO_POSE`:

```bash
ros2 param set /mavros/local_position tf.send true
```

`map` is PX4's EKF2 origin, which EKF2 fixes at the first valid GPS after boot —
wherever the FCU last powered up, not the takeoff point. **Power-cycle the FCU on
site, after RTK is fixed**, so the origin lands where you start. Booting before
RTK converges anchors it to a plain 3D fix and the position shifts by metres when
it fixes.

```bash
ros2 run tf2_ros tf2_echo map base_link   # ~0,0,0 at the start point
```

A large reading here means the grid does not contain the aircraft.

## ⚙️ Configuration

All node parameters live in a single ROS 2 params file, `config/exploration_params.yaml` (installed to `share/exploration_v2/config/exploration_params.yaml`). The launch file loads it for every node, so the `declare_parameter` defaults in the Python sources are never the effective values at runtime.

| Section | Applies to | Contents |
| :--- | :--- | :--- |
| `/**` | all nodes | camera frame & topics, pose source, `h_ground`, `max_projection_dist`, `active_domain_radius`, `marker_scale`, grid bounds and `delta_grid` |
| `camera_extrinsics_publisher` | static TF `base_link -> camera_link` | camera mounting offset and tilt |
| `vlm_node` | `vlm_node` | `max_bbox_area_ratio` |
| `semantic_belief_tracker_node` | `semantic_belief_tracker_node` | Bayes filter gains (`nu`, `gamma_context`, `alpha_obs`, `beta_absent`) |
| `planner_node` | `planner_node` | `flight_altitude`, `frontier_spacing`, IG weights, `gamma`, movement limits, `bounds_*`, `debug_planner` |
| `visual_servoing_node` | `visual_servoing_node` | altitude limits, `standoff_distance`, `x_dist`, depth validity range |
| `mission_manager` | `mission_manager_node` (not launched) | clues file, action name, report options |

> **NOTE:** pose comes from MAVROS (`odom_topic: /mavros/local_position/odom`, `world_frame: map`, `odom_frame: base_link`). Geometry is set for a 20 × 20 m box at x −10..10 East, y −5..15 North — centre 5 m north of the `map` origin. Resizing it means changing `grid_*` in three places — `/**`, `planner_node` (`bounds_*`) and `mission_manager` — plus the altitudes and radii listed in the geometry section header.

To override a single parameter without editing the file:
```bash
ros2 run exploration_v2 planner_node --ros-args --params-file install/exploration_v2/share/exploration_v2/config/exploration_params.yaml -p flight_altitude:=3.0
```

Command-line `-p` wins over `--params-file`, and inside the file a node-specific section wins over `/**`.

`w_semantic` and `w_reobs` are in both the YAML and the launch arguments. The launch file passes them after the params file, so the launch argument always wins — including at its default. Change the launch argument to run an ablation; under `ros2 run` or tmuxinator the YAML value is the effective one.

## 🚀 Running Visual Search Missions

Missions are configured through the JSON file `vlm_clues.json` located in `config/` (installed under `share/exploration_v2/config/vlm_clues.json`):

```bash
ros2 run exploration_v2 mission_manager_node
```

`mission_manager_node` will automatically:
- Load the target clues from `config/vlm_clues.json`.
- Connect to `/start_mission` action server provided by `planner_node`.
- Send goals to `planner_node` with the primary object and contextual surroundings.
- Save mission outcome summaries and trajectory logs under `mission_reports/`.

---

## 📋 Exposed Services & Actions Overview

| Service / Action | Interface Type | Provider Node | Description |
| :--- | :--- | :--- | :--- |
| `/reset_semantic_belief` | `interfaces/srv/ResetBelief` | `semantic_belief_tracker_node` | Resets Bayesian belief grid and loads episode VLM parameters |
| `/reset_coverage_belief` | `interfaces/srv/ResetBelief` | `coverage_map_node` | Resets cumulative coverage grid and updates map bounds |
| `/generate_waypoints` | `interfaces/srv/GenerateWaypoints` | `coverage_map_node` | Generates candidate waypoints along frontier boundary $C_t$ and around semantic clues |
| `/evaluate_waypoints` | `interfaces/srv/EvaluateWaypoints` | `semantic_belief_tracker_node` | Evaluates Information Gain ($\text{IG}_{\text{geom}}$, $\text{IG}_{\text{sem}}$) and cost score $J(w)$ |
| `/acquire_observation` | `interfaces/srv/AcquireObservation` | `coverage_map_node` | Fuses current camera frustum ground footprint into cumulative coverage map |
| `/reset_coverage` | `std_srvs/srv/Trigger` | `coverage_map_node` | Resets cumulative ground coverage map and clears RViz markers |
| `/update_posterior` | `interfaces/srv/UpdatePosterior` | `semantic_belief_tracker_node` | Runs one full Sense step (VLM detection, Monte Carlo ray-casting, log-space Bayes update) |
| `/detect_semantic_clues` | `interfaces/srv/DetectSemantics` | `vlm_node` | Detects 2D bounding boxes for target and context queries via Qwen-VL |
| `/verify_target` | `interfaces/srv/VerifyTarget` | `vlm_node` | Deterministic close-range target verification |
| `/stop_mission` | `std_srvs/srv/Trigger` | `planner_node` | Requests immediate cancellation of active exploration mission |
| `/start_mission` | `interfaces/action/StartMission` | `planner_node` | Action Server to execute search mission for a target object |

---

## 💻 Terminal Usage (`ros2 service call`)

### `/reset_semantic_belief`
Resets the belief tracker grid for a new target instruction and map. The grid
bounds in the call override the YAML, so keep them equal to the configured box.
```bash
ros2 service call /reset_semantic_belief interfaces/srv/ResetBelief "{
  primary_object: 'lion plush',
  full_instruction: 'lion plush',
  surroundings_json: '[\"black backpack\"]',
  sigmas_json: '[3.0]',
  d_stars_json: '[1.0]',
  map_name: 'flight_test',
  grid_x_min: -10.0,
  grid_x_max: 10.0,
  grid_y_min: -5.0,
  grid_y_max: 15.0
}"
```

---

## 🎯 Executing the Action Server (`/start_mission`)

To trigger a full autonomous search mission using the ROS 2 Action Server:

```bash
ros2 action send_goal --feedback /start_mission interfaces/action/StartMission "{
  primary_object: 'lion plush',
  object_description: '',
  full_instruction: 'lion plush',
  surroundings_json: '[\"black backpack\"]',
  sigmas_json: '[3.0]',
  d_stars_json: '[1.0]',
  map_name: 'flight_test',
  grid_x_min: -10.0,
  grid_x_max: 10.0,
  grid_y_min: -5.0,
  grid_y_max: 15.0
}"
```

## Ablation
1. Only geometric gain:
    ```bash
    ros2 launch exploration_v2 exploration_mission.launch.py dataset_filter:="dataset_Venice" w_semantic:=0.0 w_reobs:=0.0
    ```
    ```bash
    ros2 run exploration_v2 mission_manager_node --ros-args -p report_dir:="ObjectNavEval/ablation/geometric_semantic"
    ```

## 🕹️ Manual Piloting

The drone is flown by a human pilot. `planner_node` never commands motion: it
publishes the viewpoint it wants and blocks until the measured pose (RTK, via the
`world -> odom_frame` TF) says the drone held it for `arrival_hold_time_sec`.
There is no publisher to any MAVROS setpoint topic anywhere in the package.

| Topic | Type | Description |
| :--- | :--- | :--- |
| `/commanded_viewpoint` | `geometry_msgs/PoseStamped` | Viewpoint the pilot is being asked to fly to (transient local) |
| `/pilot_guidance` | `visualization_msgs/MarkerArray` | Acceptance volume, heading arrow and live `dxy / dz / dyaw` readout for RViz |

Arrival parameters on `planner_node`:

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `arrival_position_tolerance` | `1.00` | XY arrival radius (m); a human cannot hold much tighter outdoors |
| `arrival_altitude_tolerance` | `0.50` | Z arrival band (m) |
| `arrival_yaw_tolerance_deg` | `20.0` | Yaw band (deg); set `180.0` to ignore heading |
| `arrival_hold_time_sec` | `1.0` | Time the drone must stay inside tolerance |
| `takeoff_timeout_sec` | `120.0` | Deadline for the pilot to reach `flight_altitude` |

Close-range verification holds indefinitely (`viewpoint_timeout_sec: 0.0`). Two
things end a hold: losing the pose for `no_pose_timeout_sec` (10 s), and

```bash
ros2 service call /abort_servoing std_srvs/srv/Trigger
```

`/stop_mission` cannot reach a hold — the planner calls verification
synchronously and is parked inside it.

Exploration steps do *not* hold. If the pilot misses one, the planner observes
where the drone actually is and continues; the belief update uses the measured
pose. Servoing holds because its geometry assumes the drone is at the commanded
viewpoint.

`ManualPilotMonitor` (`pilot.py`) owns the wait and the guidance publishers, used
in `send_move_command` / `send_rotate_command` / `wait_for_flight_altitude`.
Those are the seam to reimplement if AeroStack2 replaces the pilot.

The camera pose is looked up at the frame's own timestamp, so an unresolvable
frame is skipped rather than projected with a stale pose:

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `vlm_frame_delay_tolerance` | `1.0` | Max age (s) of a frame before the VLM update is skipped |
| `tf_timeout_sec` | `0.1` | How long TF may wait for the pose at that timestamp; raise it if RTK TF lags the camera |

### NOTES

- Change the mission in vlm_clues.json
- Pose comes from MAVROS over TELEM2 (`/dev/ttyTHS1`, 921600), which publishes `map -> base_link` directly. There is no `outdoor_localization` package and none is needed. `tf.send` defaults to false and must be set on the `local_position` sub-node after every MAVROS start, or the topic publishes while TF stays empty.
- Coverage builds geometric, grid bayes filter is the semantic gridmap instanced by semanticbelieftracker, semantic belief, visual servoing. 