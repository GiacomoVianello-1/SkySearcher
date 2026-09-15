import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

MOTIVE_SERVER = '192.168.0.100'
MOTIVE_PORT   = 3883


def generate_launch_description():
    pkg = get_package_share_directory('ground_station_bringup')
    rviz_config = os.path.join(pkg, 'config', 'viz.rviz')
    tf_broadcaster_config = os.path.join(pkg, 'config', 'tf_broadcaster.yaml')

    # OptiTrack is indoor only. Outdoors the pose comes from RTK via MAVROS on
    # the Jetson, and the ground station only runs RViz.
    use_mocap = LaunchConfiguration('use_mocap')

    return LaunchDescription([

        DeclareLaunchArgument(
            'use_mocap',
            default_value='false',
            description='Start the OptiTrack VRPN client and its TF broadcaster (indoor arena only)',
        ),

        # --- VRPN MoCap client ---
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource([
                get_package_share_directory('vrpn_mocap'),
                '/launch/client.launch.yaml'
            ]),
            launch_arguments={
                'server': MOTIVE_SERVER,
                'port': str(MOTIVE_PORT),
                'sensor_data_qos': 'false',
            }.items(),
            condition=IfCondition(use_mocap),
        ),

        # --- VRPN TF broadcaster ---
        Node(
            package='ground_station_bringup',
            executable='tf_broadcaster.py',
            name='tf_broadcaster',
            parameters=[tf_broadcaster_config],
            output='screen',
            condition=IfCondition(use_mocap),
        ),

        # --- RViz2 ---
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
        ),
    ])