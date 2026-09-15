# 💻 Ground Station
The scripts contained in this folder are meant to be exectued by the ground station when communicating with the agent.

## 🐳 Usage via Docker

The ground station runs RViz and the visualizer, which need OpenGL but not CUDA, so no NVIDIA container runtime is required.

Now you shall be able to:
1. **Build the Docker Image** for the ground station:
    ```bash
    cd ground_station
    docker build -f docker/Dockerfile.ground_station -t robotics_ia:ground_station .
    ```
2. Bring up the **container** in detached mode:
    ```bash
    cd ground_station
    docker compose up -d
    ```
3. Attach an **interactive shell** to the running container:
    ```bash
    docker exec -it ground_station-robotics_ia_ground_station-1 bash
    ```
    (Ensure your user is in the `docker` group to run this without `sudo`).
4. Launch the `ground_station_bringup` package:
    ```bash
    ros2 launch ground_station_bringup ground_station.launch.py
    ```
5. **Destroy** the container:
    ```bash
    docker compose down
    ```

## Check the Communication

Before testing the overall system, it is strongly recommended to test the communication between the Jetson and the ground station. To do so we can:
1. Run in the Jetson container:
    ```bash
    ros2 run demo_nodes_cpp talker
    ```
2. Run in the Ground Station container:
    ```bash
    ros2 run demo_nodes_cpp listener
    ```
If the ROS DDS does not work, you will need to adjust the `cyclone_config.xml` files for both the Jetson and the ground station.

## 🤖 ROS 2 Packages

For what concerns the Jetson platform, the ROS 2 packages used in this project can be found under `ros2_ws` and are:
- [**ground_station_bringup**](/ground_station/ros2_ws/src/ground_station_bringup/): enables the OptiTrack ground-truth pose acquisition, publishes TFs  and load a custom RViz visualization.