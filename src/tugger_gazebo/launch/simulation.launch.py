import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription,
    OpaqueFunction, RegisterEventHandler, SetEnvironmentVariable)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from ament_index_python.packages import get_package_prefix, get_package_share_directory


# A busy machine can take longer than the spawner's default 5 s to start the
# control loop. A timed-out switch leaves that controller inactive for the
# whole session (seen with lift_controller), so allow more time.
SWITCH_TIMEOUT_S = '30'


def generate_launch_description():
    return LaunchDescription([
        # Empty keeps the normal GUI. Validation runs pass
        # '-s --headless-rendering' so cameras render without a window.
        DeclareLaunchArgument('gz_extra_args', default_value=''),
        # Another world file (e.g. a station with a worn marker for validation).
        # Empty = the normal warehouse world.
        DeclareLaunchArgument('world', default_value=''),
        # Fault testing only (tools/fault_testing): 'true' publishes the
        # laser scanners and the IMU under /sim_raw/..., and the fault relay
        # passes them on to the normal topics, with a chosen fault applied.
        DeclareLaunchArgument('sensor_faults', default_value='false'),
        # Fault testing only: the simulated wheel radius (odometry keeps 0.055).
        DeclareLaunchArgument('sim_wheel_radius', default_value='0.055'),
        OpaqueFunction(function=_simulation),
    ])


def _simulation(context):
    raw = '/sim_raw' if LaunchConfiguration('sensor_faults').perform(context) == 'true' else ''

    description_pkg = get_package_share_directory(
        'tugger_description'
    )

    gazebo_pkg = get_package_share_directory(
        'tugger_gazebo'
    )

    ros_gz_sim_pkg = get_package_share_directory(
        'ros_gz_sim'
    )

    xacro_file = os.path.join(
        description_pkg,
        'urdf',
        'tugger.urdf.xacro'
    )

    world_file = LaunchConfiguration('world').perform(context) or os.path.join(
        gazebo_pkg,
        'worlds',
        'tugger_warehouse.sdf'
    )

    robot_description = ParameterValue(
        Command([
            'xacro ',
            xacro_file
        ]),
        value_type=str
    )
    # Fault testing only: simulated tyres of another radius (e.g. compressed
    # under load). Only the simulated contact sphere changes; the description
    # file and everything the robot computes keep the real 0.055 m wheel.
    sim_wheel_radius = LaunchConfiguration('sim_wheel_radius').perform(context)
    if float(sim_wheel_radius) != 0.055:
        import subprocess
        urdf = subprocess.run(['xacro', xacro_file], check=True, capture_output=True,
                              text=True).stdout
        contact = '<sphere radius="0.055"/>'
        assert urdf.count(contact) == 4, 'expected four wheel contact spheres'
        robot_description = urdf.replace(contact, f'<sphere radius="{sim_wheel_radius}"/>')

    # Allow Gazebo to resolve package://tugger_description/... mesh URIs.
    gazebo_resource_path = os.pathsep.join([
        os.path.dirname(description_pkg),
        # `model://tugger_gazebo/...` needs the directory that contains the
        # package directory, not the package directory itself.
        os.path.dirname(gazebo_pkg),
        gazebo_pkg,
    ])

    set_gazebo_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=[
            gazebo_resource_path,
            os.pathsep,
            os.environ.get('GZ_SIM_RESOURCE_PATH', '')
        ]
    )

    set_gazebo_plugin_path = SetEnvironmentVariable(
        name='GZ_SIM_SYSTEM_PLUGIN_PATH',
        value=[
            os.path.join(get_package_prefix('tugger_gazebo'), 'lib'),
            os.pathsep,
            os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', '')
        ]
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True
        }]
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                ros_gz_sim_pkg,
                'launch',
                'gz_sim.launch.py'
            )
        ),
        launch_arguments={
            'gz_args': [f'-r {world_file} ', LaunchConfiguration('gz_extra_args')]
        }.items()
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'autonomous_tugger',
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.15'
        ],
        output='screen'
    )

    # Bridge Gazebo simulation time to ROS 2 /clock.
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='clock_bridge',
        arguments=[
            '/world/tugger_test/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
        ],
        remappings=[
            ('/world/tugger_test/clock', '/clock')
        ],
        output='screen'
    )

    imu_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='imu_bridge',
        arguments=[
            '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU'
        ],
        remappings=[
            ('/imu', raw + '/imu/data')
        ],
        output='screen'
    )

    # Bridge both Gazebo LiDARs to ROS 2.
    lidar_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='lidar_bridge',
        arguments=[
            '/front_left/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/rear_right/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan'
        ],
        remappings=[
            ('/front_left/scan', raw + '/front_left/scan'),
            ('/rear_right/scan', raw + '/rear_right/scan'),
        ],
        output='screen'
    )

    # Gazebo publishes contact sensors on their scoped topic (the <topic>
    # element is ignored); bridge it and remap it to the bumper topic.
    bumper_topic = ('/world/tugger_test/model/autonomous_tugger/link/base_footprint/'
                    'sensor/bumper/contact')
    bumper_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='bumper_bridge',
        arguments=[bumper_topic + '@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts'],
        remappings=[(bumper_topic, '/tugger/bumper/contacts')],
        output='screen'
    )

    # Bridge the depth, downward and top QR cameras to ROS 2.
    camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='camera_bridge',
        arguments=[
            '/front_depth/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
            '/rear_depth/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
            '/downward_camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/top_qr_camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/downward_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo'
        ],
        output='screen'
    )

    # Colour cameras for obstacle recognition. Lazy: Gazebo renders them only
    # while a ROS node subscribes, so they cost nothing when unused.
    recognition_camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='recognition_camera_bridge',
        arguments=[
            '/front_rgb/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/rear_rgb/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
        ],
        parameters=[{'lazy': True}],
        output='screen'
    )

    # The detachable-joint Gazebo systems use Empty messages for cargo pickup
    # and release.  The RQt panel publishes the ROS endpoints below.
    cargo_attachment_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='cargo_attachment_bridge',
        arguments=[
            '/cargo/station_0/attach@std_msgs/msg/Empty]gz.msgs.Empty',
            '/cargo/station_0/detach@std_msgs/msg/Empty]gz.msgs.Empty',
            '/cargo/station_3/attach@std_msgs/msg/Empty]gz.msgs.Empty',
            '/cargo/station_3/detach@std_msgs/msg/Empty]gz.msgs.Empty',
            # Phase B4: cargo state and the collision scene come from the
            # robot's own estimate (cargo_estimator), not from the simulator.
            # The detachable joint stays as the simulated lift actuator.
        ],
        output='screen'
    )

    # Automatically load and activate ROS 2 controllers.
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager',
            '--switch-timeout', SWITCH_TIMEOUT_S
        ],
        output='screen'
    )

    steering_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'steering_controller',
            '--controller-manager', '/controller_manager',
            '--switch-timeout', SWITCH_TIMEOUT_S,
            '--param-file',
            os.path.join(
                gazebo_pkg,
                'config',
                'controllers.yaml'
            )
        ],
        output='screen'
    )

    traction_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'traction_controller',
            '--controller-manager', '/controller_manager',
            '--switch-timeout', SWITCH_TIMEOUT_S,
            '--param-file',
            os.path.join(
                gazebo_pkg,
                'config',
                'controllers.yaml'
            )
        ],
        output='screen'
    )


    lift_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'lift_controller',
            '--controller-manager', '/controller_manager',
            '--switch-timeout', SWITCH_TIMEOUT_S,
            '--param-file',
            os.path.join(
                gazebo_pkg,
                'config',
                'controllers.yaml'
            )
        ],
        output='screen'
    )

    return [
        set_gazebo_resource_path,
        set_gazebo_plugin_path,
        robot_state_publisher,
        gazebo,
        clock_bridge,
        imu_bridge,
        lidar_bridge,
        camera_bridge,
        recognition_camera_bridge,
        cargo_attachment_bridge,
        bumper_bridge,
        spawn_robot,
        # Controllers start only once the robot exists in Gazebo.
        RegisterEventHandler(OnProcessExit(
            target_action=spawn_robot,
            on_exit=[
                joint_state_broadcaster_spawner,
                steering_controller_spawner,
                traction_controller_spawner,
                lift_controller_spawner,
            ])),
    ]
