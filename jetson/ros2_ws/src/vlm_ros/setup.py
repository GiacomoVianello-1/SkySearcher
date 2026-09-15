import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'vlm_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='giacomo',
    maintainer_email='giacomo.vianello.2@studenti.unipd.it',
    description='VLM utility package for ROS2 integration',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'vlm_node = vlm_ros.vlm_node:main',
            'test_vlm_client = vlm_ros.test_vlm_client:main',
            'query_node = vlm_ros.query_node:main',
        ],
    },
)
