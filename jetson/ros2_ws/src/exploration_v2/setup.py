import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'exploration_v2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', [f for f in glob('config/*') if os.path.isfile(f)]),
        ('share/' + package_name + '/launch', [f for f in glob('launch/*') if os.path.isfile(f)]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='giacomo',
    maintainer_email='giacomo.vianello.2@studenti.unipd.it',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'coverage_map_node = exploration_v2.coverage_map_node:main',
            'vlm_node = exploration_v2.vlm_node:main',
            'semantic_belief_tracker_node = exploration_v2.semantic_belief_tracker:main',
            'planner_node = exploration_v2.planner:main',
            'mission_manager_node = exploration_v2.mission_manager:main',
            'visual_servoing_node = exploration_v2.visual_servoing_node:main',
        ],
    },
)
