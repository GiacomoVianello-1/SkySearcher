<div align="center">

# SkySearcher: High-Altitude Onboard Vision-Language Reasoning for Aerial Query Localization

</div>

This repository contains a full-stack ROS 2 Humble framework designed for the **NVIDIA Jetson Orin Nano**. It integrates Vision Language Models (VLMs) like **Qwen** and **SmolVLM** with an **Intel RealSense D435i** depth camera to perform real-time semantic object detection and calculate 3D spatial coordinates for robotic navigation.

## 📄 Abstract
Recent advances in **Vision-Language Models** (VLMs) have improved semantic and visual reasoning, enabling UAVs to search for open-vocabulary queries from user instructions in unstructured environments at high altitude. However, existing systems mainly follow low-altitude strategies, focusing on close-range iterative exploration, requiring storage-intensive dense 3D metric-semantic reconstructions or relying on large cloud-serviced VLMs assuming constant internet access. We propose **SkySearcher**, an aerial semantic search navigation method designed for large outdoor scenes. **SkySearcher** integrates onboard VLM reasoning with a lightweight probabilistic representation that guides an information-driven exploration. The VLM extracts semantic information at high altitude about the target or cues informing about its location. This information is projected to the ground using the camera frustum instead of limited-depth sensors, actively exploiting the expansive field of view offered by elevated sensing. The information regarding the presence or absence of semantic cues is then integrated into a probabilistic map by modeling the spatial likelihood of the target, maintaining a task-driven memory for the mission and guiding the exploration towards semantically promising regions. Extensive simulations for Aerial Object Goal Navigation on public benchmarks demonstrate that **SkySearcher** significantly outperforms state-of-the-art baselines in success rate, flight time, and navigation precision, while a deployment of **SkySearcher** running fully onboard a drone demonstrates the applicability of our approach.

## ⚙️ Installation and Usage

There are no specific system requirements since the entire project relies on **Docker** for containerization. Specifically, we have a Docker instance implementing all the embedded logic for the **quadrotor** agent and a Docker instance for the **ground station**.  

### 📜 Preliminaries on Jetson Orin Nx 16 Gb
1. Make sure to have a machine running **Ubuntu 22.04**.
2. Install and configure the Jetson board by following this [guide](https://www.jetson-ai-lab.com/tutorials/initial-setup-sdk-manager/).
3. Next, clone the **Jetson Containers** repository. Follow the instructions [here](https://github.com/dusty-nv/jetson-containers). 
4. Install **Docker** in the hardware (required). You can follow this [tutorial](https://www.jetson-ai-lab.com/tutorials/ssd-docker-setup/#docker).
5. Now you can build images and containers usec in this work.

> [!NOTE]
> If you access your Jetson remotely through SSH, you can disable the Ubuntu desktop GUI. This frees around ~800 MB that the window manager and desktop normally consume. Moreover, it’s advisable to mount swap space (typically correlated with the amount of memory on the board). Refer to this [post](https://www.jetson-ai-lab.com/tutorials/ram-optimization/).

### 🚀 Usage
Once you have completed the preliminary part, choose the correct folder, then build and run the **Docker image** of that device:
- [**Ground Station:**](/ground_station/) this is the machine receiving data from the quadrotor and sending service (or action) requests. Read carefully this [README](/ground_station/README.md).
- [**Jetson:**](/jetson/) this is the companion computer running in the quadrotor, acting as the high-level ROS 2 coordination pipeline and implementing the ObjectNav logic. Read carefully this [README](/jetson/README.md).

### Execution Flow

- If you are using the **OptiTrack system**:
   1. Create a new container and enter it:
      ```bash
      docker compose up -d
      docker exec -it ground_station-robotics_ia_ground_station-1 bash
      ```
   2. Launch the package that handles the conversions between Optitrack ground-thruth and ROS messages (with TFs):
      ```bash
      cd ros2_ws
      source install/setup.bash
      ros2 launch ground_station_bringup ground_station.launch.py
      ```
      > **NOTE:** Remeber to run `xhost +` in any terminal to enable the GUIs usage from the container.
   3. Via SSH, create, run, and enter the Jetson container:
      ```bash
      docker compose up -d
      docker attach jetson-robotics_ia-1 
      ```
      This will open a Tmux session in your current terminal where you can launch nodes inside the Jetson.

## 📂 File Structure

This repository is split into two main sections: one for the onboard Jetson companion computer and one for the Ground Station.

```text
robotics_ia/
├── ground_station/            # Ground Station workspace and configuration
│   ├── docker/                # Docker deployment files for the Ground Station
│   │   └── Dockerfile.ground_station
│   ├── cyclone_config.xml     # CycloneDDS configuration
│   ├── docker-compose.yml     # Compose file to spin up the ground station container
│   ├── ros2_ws/               # Ground Station ROS 2 workspace (OptiTrack bridge, etc.)
│   └── visualizer/            # Rerun visualizer & plotting scripts for mission logs
└── jetson/                    # Onboard Jetson Orin Nano companion computer files
    ├── docker/                # Onboard Dockerfiles (Jetson VLM container, Turtlebot bridge)
    ├── cyclone_config.xml     # CycloneDDS configuration (onboard)
    ├── docker-compose.yml     # Onboard main orchestration
    ├── ros2_ws/               # Onboard ROS 2 workspace (ObjectNav, VLM node, coverage map)
    └── test_images/           # Images for testing VLM and detection pipeline
```

## 📊 Visualization

We utilize [Rerun](https://rerun.io/) as our primary 3D live visualization suite. It reads ROS 2 bags recorded during real-world runs and correlates the temporal messages (such as camera streams, TF transforms, and occupancy grids) with the static JSON mission reports generated by the planning nodes.

For setup requirements, recording guides, and instructions on how to run the visualizer (including the step-by-step playback mode), please check the [visualizer README](./ground_station/visualizer/README.md).

Alternatively, you can launch the RViz configuration in `ground_station/ros2_ws/src/config/viz.rviz`.


## Hardware and Software details

### Hardware
Everything runs on an **NVIDIA Jetson Orin NX** (16GB RAM unified memory), with JetPack 6.2 (L4T version: 36.5), inside a container build from a specific Docker Image.

The depth camera is an **Intel RealSense D435i**, a stereo depth camera with an integrated IMU. It provides synchronized RGB and depth streams, which you use to fuse 2D semantic detections into 3D spatial coordinates.

The air-frame is an Holybro X500, and the flight controller is a Pixhawk 4, running the PX4 autopilot firmware. You can find more details about the firmware [here](https://docs.px4.io/main/).

### Software Stack
- **Docker** based on `dustynv/l4t-pytorch:r36.4.0` — NVIDIA's official L4T image with PyTorch pre-built for Jetson ARM64
- **ROS2 Humble** on Ubuntu 22.04 Jammy (the only ROS2 distro compatible with L4T 36.x)
- **Qwen3.5-0.8B** via HuggingFace transformers — the largest VLM that fits in 8GB with the rest of the system running
- **librealsense + realsense-ros** for camera integration