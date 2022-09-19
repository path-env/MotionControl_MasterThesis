"""
Setup for carla_bci_control
"""
import os
from glob import glob
ROS_VERSION = int(os.environ['ROS_VERSION'])

if ROS_VERSION == 1:
    from distutils.core import setup
    from catkin_pkg.python_setup import generate_distutils_setup

    d = generate_distutils_setup(
        packages=['carla_bci_control'],
        package_dir={'': 'src'}
    )

    setup(**d)

elif ROS_VERSION == 2:
    from setuptools import setup

    package_name = 'carla_bci_control'
    setup(
        name=package_name,
        version='0.0.0',
        packages=['src/' + package_name],
        data_files=[
            ('share/ament_index/resource_index/packages',
             ['resource/' + package_name]),
            ('share/' + package_name, ['package.xml']),
            ('share/' + package_name + '/config',
             ['config/objects.json']),
            (os.path.join('share', package_name), glob('launch/*.launch.py'))
        ],
        install_requires=['setuptools'],
        zip_safe=True,
        maintainer='Mangal ',
        maintainer_email='xxx@gmail.com',
        description='CARLA BCI control for ROS2 bridge',
        license='MIT',
        tests_require=['pytest'],
        entry_points={
            'console_scripts': [
                'carla_bci_control = src.carla_bci_control.carla_bci_control:main',
                'set_initial_pose = src.carla_bci_control.set_initial_pose:main'
            ],
        },
    )
