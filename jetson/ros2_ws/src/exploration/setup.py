from setuptools import find_packages, setup

package_name = 'exploration'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/exploration_mission.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='giacomo',
    maintainer_email='giacomo.vianello.2@studenti.unipd.it',
    description='Exploration package for ROS 2',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'coverage_map_node = exploration.coverage_map_node:main',
            'semantic_belief_tracker_node = exploration.semantic_belief_tracker:main',
            'planner_node = exploration.planner:main',
            'mission_manager_node = exploration.mission_manager:main',
            'vlm_node = exploration.vlm_node:main',
        ],
    },
)
