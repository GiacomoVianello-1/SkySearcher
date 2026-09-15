import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription

def generate_launch_description():
    package_name = 'exploration_v2'

    pkg_dir = get_package_share_directory(package_name)

    # All node parameters live here; see config/exploration_params.yaml.
    params_file = os.path.join(pkg_dir, 'config', 'exploration_params.yaml')

    # Joins the MAVROS chain (map -> base_link) to the RealSense chain; without it the
    # TF tree is two islands and every world -> camera lookup fails. Kept in its own
    # launch file so the tmuxinator panes can start it without the rest of the stack.
    camera_extrinsics_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, 'launch', 'camera_tf.launch.py')
        )
    )

    # Ablation: passed after params_file below, so these win over the YAML values.
    w_semantic_arg = DeclareLaunchArgument(
        'w_semantic',
        default_value='15.0',
        description='Weight for semantic information gain'
    )
    w_reobs_arg = DeclareLaunchArgument(
        'w_reobs',
        default_value='0.001',
        description='Weight for re-observation information gain'
    )

    # --- Nodes ---

    qwen_vlm_node = Node(
        package=package_name,
        executable='vlm_node',
        name='vlm_node',
        output='screen',
        parameters=[params_file]
    )

    coverage_map_node = Node(
        package=package_name,
        executable='coverage_map_node',
        name='coverage_map_node',
        output='screen',
        parameters=[params_file]
    )

    semantic_belief_tracker_node = Node(
        package=package_name,
        executable='semantic_belief_tracker_node',
        name='semantic_belief_tracker_node',
        output='screen',
        parameters=[params_file]
    )

    exploration_planner_node = Node(
        package=package_name,
        executable='planner_node',
        name='planner_node',
        output='screen',
        parameters=[
            params_file,
            {
                'w_semantic': LaunchConfiguration('w_semantic'),
                'w_reobs': LaunchConfiguration('w_reobs'),
            },
        ]
    )

    visual_servoing_node = Node(
        package=package_name,
        executable='visual_servoing_node',
        name='visual_servoing_node',
        output='screen',
        parameters=[params_file]
    )
    
    return LaunchDescription([
        w_semantic_arg,
        w_reobs_arg,
        camera_extrinsics_node,
        qwen_vlm_node,
        coverage_map_node,
        semantic_belief_tracker_node,
        visual_servoing_node,
        exploration_planner_node,
    ])