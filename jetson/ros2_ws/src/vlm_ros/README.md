# VLM ROS PACKAGE

This package is meant for handling the VLM via HugginFace inside ROS2. It contains:
```bash
vlm_ros/
├── vlm_node        # VLM Server (exposes DetectSemantics srv)
├── query_node      # VLM Client (exposes AcquireImage srv)
└── test_vlm_client # Script test that calls the VLM onto a pre-taken image
```
An example of its functionalities is provided below. 

<div align="center">
  <img src="../../saved_images/VLM_example.png" alt="VLM Example"/>
</div>

## Usage

1. In one terminal launch the RealSense2 ROS Wrapper:
    ```bash
    ros2 launch realsense2_camera rs_launch.py
    ```
2. In a second terminal launch the package:
    ```bash
    ros2 launch vlm_ros vlm_launch.launch.py 
    ```
    This will start the Server node (`vlm_node`), the Client node (`query_node`), and the RViz visualization node. Inside the launch file (`vlm_launch.launch.py `) you can configure the maximum size of the VLM output bounding boxes (`max_bbox_area_ratio`) and the target query for the VLM inspection (`target_query`).
3. In a third terminal you can try the Object detection capabilities by calling
    ```bash
    ros2 service call /acquire_image std_srvs/srv/Trigger "{}"
    ```
