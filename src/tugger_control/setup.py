from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'tugger_control'
route_files = glob('../../tools/v1_manhattan_bezier/routes/*.json')
docking_files = glob('../../tools/v1_manhattan_bezier/phase14/routes_refined/*__REFINED.json')
entrance_candidates = glob('../../tools/mission_approaches/entrance_candidates.json')

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml', 'plugin.xml']
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml') + glob('config/*.json')
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
        (
            os.path.join('share', package_name, 'routes'),
            route_files,
        ),
        (
            os.path.join('share', package_name, 'docking'),
            docking_files + entrance_candidates,
        ),
    ],
    install_requires=['setuptools'],
    extras_require={'test': ['pytest']},
    zip_safe=True,
    maintainer='osama',
    maintainer_email='osama@example.com',
    description='Vehicle control for the autonomous 4WS tugger AMR',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'four_ws_controller = tugger_control.four_ws_controller:main',
            'four_ws_odometry = tugger_control.four_ws_odometry:main',
            'qr_docking_vision = tugger_control.qr_docking_vision:main',
            'sim_motion_gate = tugger_control.sim_motion_gate:main',
            'safety_monitor = tugger_control.safety_monitor:main',
            'scan_merger = tugger_control.scan_merger:main',
            'cargo_estimator = tugger_control.cargo_estimator:main',
            'cargo_id_reader = tugger_control.cargo_id_reader:main',
            'marker_localization = tugger_control.marker_localization:main',
            'docking_precision_controller = tugger_control.docking_precision_controller:main',
            'imu_orientation_plotter = tugger_control.imu_orientation_plotter:main',
            'nav2_startup_guard = tugger_control.nav2_startup_guard:main',
        ],
        'rqt_gui_py.plugin': [
            'tugger_control = tugger_control.operator_panel:OperatorPanel',
        ],
    },
)
