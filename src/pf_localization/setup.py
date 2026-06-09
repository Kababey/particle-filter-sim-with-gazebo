import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'pf_localization'


def recursive_data_files(src_dir, install_root):
    """Mirror a directory tree into share/<pkg>/<install_root> for ament install."""
    entries = []
    for path in glob(os.path.join(src_dir, '**', '*'), recursive=True):
        if os.path.isfile(path):
            rel = os.path.relpath(os.path.dirname(path), src_dir)
            dest = os.path.join('share', package_name, install_root, rel)
            entries.append((dest, [path]))
    return entries


data_files = [
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    (os.path.join('share', package_name, 'config'), glob('config/*')),
    (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
]
# Models and materials are nested trees (model.sdf, textures, etc.)
data_files += recursive_data_files('models', 'models')
data_files += recursive_data_files('materials', 'materials')

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eg',
    maintainer_email='kababey111@gmail.com',
    description='Multi-hypothesis particle filter localization with AR tags in Gazebo.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'particle_filter_node = pf_localization.particle_filter_node:main',
            'tag_detector_node = pf_localization.tag_detector_node:main',
            'sim_detector_node = pf_localization.sim_detector_node:main',
            'odom_noise_node = pf_localization.odom_noise_node:main',
            'viz_node = pf_localization.viz_node:main',
            'generate_world = pf_localization.generate_world:main',
        ],
    },
)
