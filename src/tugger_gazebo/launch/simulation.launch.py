import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

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

    world_file = os.path.join(
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
            'gz_args': f'-r {world_file}'
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
            ('/imu', '/imu/data')
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
        output='screen'
    )

    # Bridge front and rear RGB cameras to ROS 2.
    camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='camera_bridge',
        arguments=[
            '/front_camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/front_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/rear_camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/rear_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/downward_camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/downward_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo'
        ],
        output='screen'
    )

    # The detachable-joint Gazebo systems use Empty messages for cargo pickup
    # and release.  The RQt panel publishes the ROS endpoints below.
    cargo_attachment_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='cargo_attachment_bridge',
        arguments=[
            '/cargo/station_0/attach@std_msgs/msg/Empty[gz.msgs.Empty',
            '/cargo/station_0/detach@std_msgs/msg/Empty[gz.msgs.Empty',
            '/cargo/station_3/attach@std_msgs/msg/Empty[gz.msgs.Empty',
            '/cargo/station_3/detach@std_msgs/msg/Empty[gz.msgs.Empty',
        ],
        output='screen'
    )

    # Automatically load and activate ROS 2 controllers.
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager'
        ],
        output='screen'
    )

    steering_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'steering_controller',
            '--controller-manager', '/controller_manager',
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
            '--param-file',
            os.path.join(
                gazebo_pkg,
                'config',
                'controllers.yaml'
            )
        ],
        output='screen'
    )

    return LaunchDescription([
        set_gazebo_resource_path,
        robot_state_publisher,
        gazebo,
        clock_bridge,
        imu_bridge,
        lidar_bridge,
        camera_bridge,
        cargo_attachment_bridge,
        spawn_robot,
        joint_state_broadcaster_spawner,
        steering_controller_spawner,
        traction_controller_spawner,
        lift_controller_spawner
    ])
