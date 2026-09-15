# 🚁 Jetson
The scripts contained in this folder are meant to be exectued by the Jetson board mounted on the agent when communicating with the ground station.

## 🐳 Usage via Docker 

Before building the main image, you first have to **configure your environment:**
- The workspace is mounted relative to `docker-compose.yml`, so the repository works from any checkout path. The remaining absolute mounts (`/home/jetson/.Xauthority`, `/home/jetson/.cache/huggingface`, `/home/jetson/jetson-containers/data`) assume the Jetson user is `jetson` — update them if yours differs.
- Update the `HF_TOKEN` environment variable with your personal Hugging Face access token to download the VLM weights.
- (*Optional*) Set your `GOOGLE_API_KEY` for cloud inference.

Once everything is properly configured, you can:
1. **Build the Docker image:**
    ```bash
    docker build -f docker/Dockerfile.jetson -t robotics_ia:jetson .
    ```
2. **Start the Container:**

    Bring up the container in detached mode. The `docker-compose.yml` is configured to grant privileged access to USB and I2C ports.
    ```bash
    cd jetson
    docker compose up -d
    ```
3. **Attach an interactive shell to the running container:**
    ```bash
    docker exec -it jetson-robotics_ia-robotics_ia-1 bash
    ```
    (Ensure your user is in the `docker` group to run this without `sudo`).
4. **Kill the running container:**
    ```bash
    docker compose down
    ```
    This will terminate the current container, but all the files inside it won't be cancelled because we're mounting the ROS 2 workspace as a persistent volume.

## 🤖 ROS 2 Packages

For what concerns the Jetson platform, the ROS 2 packages used in this project can be found under `ros2_ws` and are:
- [**exploration**](/jetson/ros2_ws/src/exploration/): Proposed ObjectNav probabilistic framework.
- [**interfaces**](/jetson/ros2_ws/src/interfaces/): Actions, services, and messages definitions. 
- [**outdoor_localization**](/jetson/ros2_ws/src/outdoor_localization/): Manages GPS coordinate frame connectivity (tf tree) and MAVROS launch parameters.
- [**perception**](/jetson/ros2_ws/src/perception/): TODO
- [**tb2_unizar**](/jetson/ros2_ws/src/tb2_unizar/): TODO
- [**vlm_local_package**](/jetson/ros2_ws/src/vlm_local_package/): TODO
- [**vlm_ros**](/jetson/ros2_ws/src/vlm_ros/): Main ROS package devoted to VLM inference via HF transformers.

## 📸 RealSense Camera

The Dockerfile we provide correctly install the `relasense-ros` library. You can launch the camera (depth stream aligned to the color stream) via:
```bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```
If you don't need the depth aligned with the color stream, you can simply use:
```bash
ros2 launch realsense2_camera rs_launch.py
```
When the `ros-wrapper` is running and `aligned_depth` is not enabled, the RealSense2 topics are:
```bash
/camera/camera/color/camera_info
/camera/camera/color/image_raw
/camera/camera/color/image_raw/compressed
/camera/camera/color/metadata
/camera/camera/depth/camera_info
/camera/camera/depth/image_rect_raw
/camera/camera/depth/image_rect_raw/compressed
/camera/camera/depth/metadata
/camera/camera/extrinsics/depth_to_color
```
Similarly, the TF frames exposd by RealSense2 ROS wrapper are:
```txt
camera_link --> camera_color_frame --> camera_color_optical_frame
camera_link --> camera_depth_frame --> camera_depth_optical_frame
```

## 🧵 Tmuxinator

In order to ease the debugging and execution we provide a **tmux template** (`robotics_ia.yml`) that offers 5 organized shells. To access the tmux session, you first need to launch the jetson container:
```bash
docker compose up -d
```
Then you can attach to that session via
```bash
docker attach jetson-robotics_ia-1
```
You can adjust the tmux setup by modifying the [`robotics_ia.yml`](/jetson/robotics_ia.yml) configuration file.

## 📡 GPS Position (via MAVROS)

In the physical hardware, we mount a Pixhawk 4 flight controller that interfaces with the **Q Ground Control** software (you can get it [here](https://docs.qgroundcontrol.com/Stable_V5.0/en/qgc-user-guide/getting_started/download_and_install.html#ubuntu)) using the PX4 Firmware. We power it via USB, directly connecting it to the Jetson USB port. 

To fly outdoor and receive position feedback, we use the Holybro M9N GPS module (connected to the Pixhawk 4). Since the Pixhawk communicate with the Jetson via serial connection, **Mavros** is used to communicate with the flight controller and to expose ROS2 topics. 

> **NOTE:** Before launching Mavros, you have to open up QGroundControl, go to *Configure* -> *Parameters* -> *Tools* -> *Reboot Vehicle*. This is because at the jetson startup the USB ports are not loaded correctly yet. So, rebooting the vehicle forces the Pixhawk to re-initialize the USB port and its serial connection. 

To run Mavros inside the container you first need to add the serial port to the Docker daemon as shown in this [`docker-compose`](/jetson/docker-compose.yml) file.

### 📍 Outdoor Localization (`outdoor_localization`)

To connect the coordinates system between GPS and the robot navigation frame, we use the `outdoor_localization` ROS 2 package.

#### What it does & Why
The package coordinates the tf tree structure for outdoor navigation, ensuring a connected chain: `map` -> `odom` -> `base_link`.
* **`map` -> `odom`**: Broadcasts a static transform with zero offset (since map and odom coincide).
* **`odom` -> `base_link`**: Published dynamically by the MAVROS `local_position` plugin.

Without this connected TF tree, downstream navigation packages (such as `exploration`) are unable to look up transforms between coordinate frames to project camera footprints and perform path planning.

#### How to run it
Inside the container, build the workspace, source it, and launch the localization:
```bash
ros2 launch outdoor_localization localization.launch.py
```
To verify that the parameters are correctly loaded and the TF transform is being published:
```bash
ros2 param get /mavros/local_position tf.send
ros2 topic echo /tf
```

#### Troubleshooting & Issues Faced
* **Global Node Name Override Bug**: Previously, specifying a custom name in the launch node definition (e.g. `name='mavros'`) passed a global renaming override to the entire MAVROS process, renaming the container, router, UAS, and all plugin subnodes to `/mavros`. Because all subnodes were renamed to `/mavros`, the wildcard parameters in `px4_config.yaml` (`/**/local_position`) could not match the node name, leading to parameters failing to load (forcing `tf.send` to default to `false`) and clashing `/mavros` nodes on the ROS graph. This was resolved by removing the `name` override from the launch node configuration to allow plugins to register under their correct namespaces (e.g. `/mavros/local_position`).
* **GPS Fix Requirement**: The dynamic `odom` -> `base_link` transform is only broadcast on `/tf` by the MAVROS `local_position` plugin once the Pixhawk has a valid GPS/position lock. If the topic is silent, verify the GPS status on QGroundControl.

## 🔧 Troubleshooting

<details>
<summary><b>RealSense problem in Jetson Orin Nano</b></summary>

The standard `apt` package `ros-humble-librealsense2` ships **x86_64 binaries only** and will not work on ARM64/Jetson. For this reason, librealsense may be compiled from source directly inside the Docker image using the `FORCE_LIBUVC=true` CMake flag. The `FORCE_LIBUVC` flag bypasses the kernel module (`librealsense2-dkms`) entirely — which is incompatible with NVIDIA's custom Tegra kernel — and instead uses the `libUVC` userspace USB backend. This is the officially recommended approach for platforms where kernel patching is not possible. More details [here](https://github.com/realsenseai/librealsense/blob/master/doc/libuvc_installation.md).

In the Dockerfile the fix consists of adding following command:
```Dockerfile
RUN git clone https://github.com/realsenseai/librealsense.git /tmp/librealsense && \
    cd /tmp/librealsense && mkdir build && cd build && \
    cmake ../ -DFORCE_LIBUVC=true -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig
```

</details>

<details>
<summary><b>Permission Denied on USB Camera</b></summary>

Ensure your host machine's Udev rules are configured, or run the container in `--privileged` mode (handled automatically by `docker-compose.yml`).

</details>

<details>
<summary><b>VLM Truncated Output</b></summary>

If the Qwen model cuts off the JSON bounding box halfway (e.g., `{"bbox_2d": [512, 17 `), it has hit the token limit. Increase `max_new_tokens` in `qwen_captioner.py` or `vlm_node.py`.

</details>

<details>
<summary><b>Out of Memory (OOM) Errors</b></summary>

The Orin (Nano, Xavier, or NX) shares RAM and VRAM. Ensure `torch.no_grad()` is active in the VLM node, and try restricting background processes on the Jetson. If you run out of memory, refer to this [post](https://www.jetson-ai-lab.com/tutorials/ram-optimization/) for some optimization.

</details>

