import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Publish base_link -> camera_link from the YAML, on its own."""
    pkg_dir = get_package_share_directory('exploration_v2')
    params_file = os.path.join(pkg_dir, 'config', 'exploration_params.yaml')

    with open(params_file) as f:
        ext = yaml.safe_load(f)['camera_extrinsics_publisher']['ros__parameters']

    camera_extrinsics_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_extrinsics_publisher',
        output='screen',
        arguments=[
            '--x', str(ext['x']), '--y', str(ext['y']), '--z', str(ext['z']),
            '--roll', str(ext['roll']), '--pitch', str(ext['pitch']), '--yaw', str(ext['yaw']),
            '--frame-id', ext['parent_frame'], '--child-frame-id', ext['child_frame'],
        ],
    )

    return LaunchDescription([camera_extrinsics_node])
