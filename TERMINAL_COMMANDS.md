# Autonomous Tugger — Terminal Commands

Run one terminal per component. Do not start duplicates.

Every terminal uses Cyclone DDS (`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`), as
Nav2 recommends. All terminals must use the same setting, including the `ros2`
checks in T8; a terminal without it may not see the others.

Start order: **Step 0 → T1 → T2 → T3 → T4 → T8 map activation → T5 → T8 checks → T6**.
T7 and T9 are optional.

## Step 0 — Start clean (every session)

Close every simulation terminal, then:

```bash
pkill -f "gz sim"; pkill -f "ros2 launch"; pkill -f rqt_gui; pkill -f rviz2; pkill -f map_server
sleep 3
ps -eo cmd | grep -E "[g]z sim|[r]os2 launch|[r]qt_gui|[m]ap_server" || echo "clean"
source /opt/ros/lyrical/setup.bash && fastdds shm clean
ros2 daemon stop
```

Continue only when it prints `clean`. The last command removes shared-memory
files left by killed ROS processes; leftovers can stop Nav2 servers from
finding each other ("action server not available").

## T1 — Gazebo

```bash
cd ~/Autonomous_Tugger_Project_V2/autonomous_tugger_ws
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH
source /opt/ros/lyrical/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export GZ_IP=127.0.0.1
export IGN_IP=127.0.0.1
ros2 launch tugger_gazebo simulation.launch.py
```

Wait until all four controllers report `activate successful`. The lift's hard
stops sit outside its working range, so it cannot latch on a limit in Gazebo
(DART bug gazebosim/gz-sim#2785); no lift command is needed at startup.

## T2 — 4WS, odometry, EKF, QR vision, docking controller, motion guard

```bash
cd ~/Autonomous_Tugger_Project_V2/autonomous_tugger_ws
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH
source /opt/ros/lyrical/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch tugger_control control.launch.py
```

## T3 — Ground-truth localization

```bash
cd ~/Autonomous_Tugger_Project_V2/autonomous_tugger_ws
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH
source /opt/ros/lyrical/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch tugger_ground_truth simulation_localization.launch.py
```

## T4 — Map server

```bash
cd ~/Autonomous_Tugger_Project_V2/autonomous_tugger_ws
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH
source /opt/ros/lyrical/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 run nav2_map_server map_server --ros-args \
  -r __node:=map_server \
  -p "yaml_filename:=$PWD/src/tugger_navigation/maps/warehouse_map.yaml" \
  -p use_sim_time:=true
```

## T5 — Nav2

```bash
cd ~/Autonomous_Tugger_Project_V2/autonomous_tugger_ws
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH
source /opt/ros/lyrical/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch tugger_navigation nav2.launch.py
```

## T6 — Operator panel

```bash
cd ~/Autonomous_Tugger_Project_V2/autonomous_tugger_ws
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH
source /opt/ros/lyrical/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 run rqt_gui rqt_gui -s tugger_control
```

## T7 — RViz (optional)

```bash
cd ~/Autonomous_Tugger_Project_V2/autonomous_tugger_ws
source /opt/ros/lyrical/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
rviz2 --ros-args -p use_sim_time:=true
```

## T8 — Map activation and checks

Map activation is required on every startup, after T4 and before T5:

```bash
cd ~/Autonomous_Tugger_Project_V2/autonomous_tugger_ws
source /opt/ros/lyrical/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 lifecycle set /map_server configure
ros2 lifecycle set /map_server activate
ros2 lifecycle get /map_server
```

It must say `active`. If `configure` says `Node not found`, wait a few seconds
and run both commands again.

After starting T5, wait about 30 s, then check Nav2 and the controllers:

```bash
ros2 lifecycle get /planner_server
ros2 lifecycle get /controller_server
ros2 lifecycle get /docking/controller_server
ros2 control list_controllers
```

All three Nav2 servers must be `active` and all four controllers `active`.
`Node not found` means T5 is not running. If a Nav2 server is `unconfigured`
(a startup race), stop T5 with Ctrl+C and start it again.

Further checks:

```bash
timeout 8s ros2 run tf2_ros tf2_echo map base_footprint
ros2 node list
ros2 topic list | grep -E 'camera|image|qr_docking|odom|collision_scene'
ros2 node info /tugger_ground_truth_bridge
ros2 topic info /tugger/cargo/station_0/ground_truth_pose -v
ros2 topic info /tugger/cargo/station_3/ground_truth_pose -v
ros2 topic echo /tugger/cargo/station_0/ground_truth_pose --once
ros2 topic echo /tugger/cargo/station_3/ground_truth_pose --once
ros2 action list | grep -E '/follow_path|/docking/follow_path'
```

## T9 — Camera viewer (optional)

```bash
cd ~/Autonomous_Tugger_Project_V2/autonomous_tugger_ws
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH
source /opt/ros/lyrical/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 run rqt_image_view rqt_image_view
```

Select `/downward_camera/image_raw`, then `/qr_docking/debug_image`.

## Required Nav2 controller check

T5 must provide exactly these two FollowPath action servers:

```bash
ros2 action list | grep -E '^/follow_path$|^/docking/follow_path$'
```

Normal navigation uses `/follow_path` and `/tugger/nav_cmd_vel`.
Precision docking uses `/docking/follow_path` and `/tugger/dock_nav_cmd_vel`.
Both command streams pass through `tugger_sim_motion_gate` before reaching
the 4WS controller.

Speed profile:

- Normal straight route: Nav2 requests up to **1.2 m/s**.
- Normal curves: the motion gate caps genuine curves at **0.6 m/s** while
  preserving the requested curvature.
- Docking approach: Nav2 requests up to **0.35 m/s**; the motion gate limit is **0.4 m/s**.
  The gate slows the robot near stand legs using its stopping-distance check.
- Final camera correction: capped at **0.05 m/s** for 10 mm settling, loaded or not.
- Loaded departure: `DepartureFollowPath` requests up to **0.45 m/s**, under the
  **0.5 m/s** gate limit, and remains subject to the loaded stopping-distance guard.

## Paths

Missions follow single-segment paths in `tools/v1_manhattan_bezier/mission_paths/`,
each running from the source dock pose to the target dock pose. After changing
the world, regenerate everything from the world file, then review and approve
the new paths before running them:

```bash
tools/regenerate_paths_from_world.sh
```

After changing speed or controller configuration, rebuild `tugger_control`
and `tugger_navigation`, then restart T2 and T5. Do not run a second launch
instance over an existing one.

```bash
cd ~/Autonomous_Tugger_Project_V2/autonomous_tugger_ws
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH
source /opt/ros/lyrical/setup.bash
colcon build --packages-select tugger_control tugger_navigation --symlink-install
```

T2 starts `qr_docking_vision`, `docking_precision_controller`, and
`tugger_sim_motion_gate`. The docking controller accepts only
`/tugger/docking/cmd_vel`, verifies fresh docking ownership, and forwards to
`/tugger/precision_cmd_vel`. Do not start these nodes separately.
new angular velocity = old angular velocity × reduced speed / requested speed
Do not start a second ground-truth bridge, map server, Nav2 launch, or operator panel.
