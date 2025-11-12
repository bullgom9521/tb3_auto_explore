from setuptools import setup


package_name = 'tb3_auto_explore'


setup(
name=package_name,
version='0.0.1',
packages=[package_name],
data_files=[
('share/ament_index/resource_index/packages', ['resource/' + package_name]),
('share/' + package_name, ['package.xml']),
('share/' + package_name + '/launch', ['launch/explore_bringup.launch.py']),
],
install_requires=['setuptools'],
zip_safe=True,
maintainer='Your Name',
maintainer_email='you@example.com',
description='Frontier exploration node driving Nav2 to explore unknown space while mapping.',
license='Apache-2.0',
tests_require=['pytest'],
entry_points={
'console_scripts': [
'frontier_explorer = tb3_auto_explore.frontier_explorer:main',
],
},
)
