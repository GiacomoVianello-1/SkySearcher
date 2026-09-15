import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    package_name = 'vlm_ros'

    pkg_share = get_package_share_directory('vlm_ros')
    rviz_config_path = os.path.join(pkg_share, 'rviz', 'viz.rviz')

    # --- Nodes ---
    vlm_server_node = Node(
        package=package_name,
        executable='vlm_node',
        name='vlm_node',
        output='screen',
        parameters=[{
            'max_bbox_area_ratio': 0.80 # Discard bounding boxes that are too large (heuristic for false positives)
        }]
    )

    vlm_client_node = Node(
        package=package_name,
        executable='query_node',
        name='query_node',
        output='screen',
        parameters=[{
            'target_query': 'toy car, robot, drone, lion plush',
            'save_images': False,          # Set to True to enable saving images
            'save_dir': 'saved_images'    # Directory where images will be saved
        }]
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
    )


    return LaunchDescription([
        vlm_server_node,
        vlm_client_node,
        #rviz # Rviz has to be launch in the ground station
    ])