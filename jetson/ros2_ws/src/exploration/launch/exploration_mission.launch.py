import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    package_name = 'exploration'

    pkg_dir = get_package_share_directory(package_name)
    instruction = "lion plush near the drone"
    
    flight_altitude = 60.0            # TODO: this doesnt make sense for the indoor case
    max_projection_distance = 200.0
        
    # The ARENA is a rectacngle of 6.12m x 5.6m, with the origin at the center centered at the mid truss on the left.
    GRID_X_MIN   = 0.0
    GRID_X_MAX   =  6.12
    GRID_Y_MIN   = -2.8
    GRID_Y_MAX   =  2.8
    GRID_DELTA   =  0.1

    # --- Nodes ---
    qwen_vlm_node = Node(
        package=package_name,
        executable='vlm_node',
        name='vlm_node',
        output='screen',
        parameters=[{
            'max_bbox_area_ratio': 0.80 # Discard bounding boxes that are too large (heuristic for false positives)
        }]
    )

    coverage_map_node = Node(
        package=package_name,
        executable='coverage_map_node',
        name='coverage_map_node',
        output='screen',
        parameters=[{
            'instruction': instruction,
            'max_projection_dist': max_projection_distance,
            'grid_x_min':  GRID_X_MIN,
            'grid_x_max':  GRID_X_MAX,
            'grid_y_min':  GRID_Y_MIN,
            'grid_y_max':  GRID_Y_MAX,
            'delta_grid':  GRID_DELTA,
        }]
    )

    semantic_belief_tracker_node = Node(
        package=package_name,
        executable='semantic_belief_tracker_node',
        name='semantic_belief_tracker_node',
        output='screen',
        parameters=[{
            'instruction': instruction,
            'h_ground':    0.0,
            'max_projection_dist': max_projection_distance,
            'grid_x_min':  GRID_X_MIN,
            'grid_x_max':  GRID_X_MAX,
            'grid_y_min':  GRID_Y_MIN,
            'grid_y_max':  GRID_Y_MAX,
            'delta_grid':  GRID_DELTA, 
            'coverage_attenuation': 0.1,
        }]
    )

    exploration_planner_node = Node(
        package=package_name,
        executable='planner_node',
        name='planner_node',
        output='screen',
        parameters=[{
            'vlm_log_path': 'vlm_scale_clues_log.json',

            'pi_0': 0.05, # Prior probability of target presence in each cell (before seeing anything)
            'gamma': 0.1,
            "flight_altitude": flight_altitude, 
            "max_projection_dist": max_projection_distance,
            'active_domain_radius': 1.0, # Make it big enough (e.g., 10 m) to cover all the arena size
            'frontier_spacing': 1.0,     # Minimum distance between waypoints [m]
            # Map's boundary
            'bounds_x_min': GRID_X_MIN, 
            'bounds_x_max': GRID_X_MAX,
            'bounds_y_min': GRID_Y_MIN,
            'bounds_y_max': GRID_Y_MAX,

            'debug_planner': True,
            'save_images': False
        }]
    )

    return LaunchDescription([
        qwen_vlm_node,
        coverage_map_node,
        semantic_belief_tracker_node,
        exploration_planner_node
    ])