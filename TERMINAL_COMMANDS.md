# Autonomous Tugger — Terminal Commands

Run one terminal per component. Do not start duplicates.

On another computer, replace `~/Autonomous_Tugger_Project_V2/autonomous_tugger_ws`
with the folder you cloned into. For install and build steps, see `README.md`.

Every terminal uses Cyclone DDS (`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`), as
Nav2 recommends. All terminals must use the same setting, including the `ros2`
checks in T8; a terminal without it may not see the others.

Every terminal also sets `CYCLONEDDS_URI`. Cyclone gives each ROS process on the
computer a numbered slot and by default offers only about 10; the full stack
fills them, so any process started later (`ros2` commands, RViz, viewers) was
refused with "Failed to find a free participant index". The setting raises the
limit to 200.

Start order: **Step 0 → T1 → T2 → T3 → T4 → T4a map activation → T4b → T5 → T8 checks → T6**.
T6b, T7, T9 and T10 are optional.

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
export CYCLONEDDS_URI='<CycloneDDS><Domain><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
export GZ_IP=127.0.0.1
export IGN_IP=127.0.0.1
ros2 launch tugger_gazebo simulation.launch.py
```

Wait until all four controllers report `activate successful`. The lift's hard
stops sit outside its working range, so it cannot latch on a limit in Gazebo
(DART bug gazebosim/gz-sim#2785); no lift command is needed at startup.

## T2 — Robot software (4WS, odometry, EKF, scan merger, safety, obstacles, QR vision, docking, motion gate)

```bash
cd ~/Autonomous_Tugger_Project_V2/autonomous_tugger_ws
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH
source /opt/ros/lyrical/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
ros2 launch tugger_control control.launch.py
```

## T3 — Ground-truth pose (evaluation only)

The robot does not use this pose. It only feeds the true position to the test
tools; the robot localizes itself in T4b.

```bash
cd ~/Autonomous_Tugger_Project_V2/autonomous_tugger_ws
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH
source /opt/ros/lyrical/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
ros2 launch tugger_ground_truth simulation_localization.launch.py
```

## T4 — Map server

```bash
cd ~/Autonomous_Tugger_Project_V2/autonomous_tugger_ws
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH
source /opt/ros/lyrical/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
ros2 run nav2_map_server map_server --ros-args \
  -r __node:=map_server \
  -p "yaml_filename:=$PWD/src/tugger_navigation/maps/warehouse_map.yaml" \
  -p use_sim_time:=true
```

## T4a — Map activation (right after T4, every startup)

Run it in a new terminal as soon as T4 is running, before T4b and T5. Without
it Nav2's global costmap never receives the map, and every path check fails
with "Path validation failed or reported an obstacle". The `CYCLONEDDS_URI`
line lets this short-lived `ros2` process join even when the running stack has
used Cyclone DDS's default of about 10 processes ("Failed to find a free
participant index").

```bash
cd ~/Autonomous_Tugger_Project_V2/autonomous_tugger_ws
source /opt/ros/lyrical/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
ros2 lifecycle set /map_server configure
ros2 lifecycle set /map_server activate
ros2 lifecycle get /map_server
```

It must say `active`. If `configure` says `Node not found`, wait a few seconds
and run both commands again.

## T4b — Localization (AMCL)

Start after the map is activated (T4a) and before T5. Without it the robot has
no `map` position and Nav2 cannot start its work.

```bash
cd ~/Autonomous_Tugger_Project_V2/autonomous_tugger_ws
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH
source /opt/ros/lyrical/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
ros2 launch tugger_navigation localization.launch.py
```

## T5 — Nav2

```bash
cd ~/Autonomous_Tugger_Project_V2/autonomous_tugger_ws
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH
source /opt/ros/lyrical/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
ros2 launch tugger_navigation nav2.launch.py
```

## T6 — Operator panel

```bash
cd ~/Autonomous_Tugger_Project_V2/autonomous_tugger_ws
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH
source /opt/ros/lyrical/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
ros2 run rqt_gui rqt_gui -s tugger_control
```

In the panel: tick **Arm**, press **Check / plan Station N** (or HOME), optionally
**ATTACH** or **DETACH**, then **RUN selected mission**. After a safety stop, clear
the cause and press **Reset safety**.

## T6b — Camera windows and spectator (optional, for watching or recording)

Start it after T6. It opens three small camera windows (QR detector, front,
rear) and locks Gazebo's view to the robot, 2 m behind its direction of travel
and 1 m up. Closing all three windows, or Ctrl+C, ends it.

```bash
cd ~/Autonomous_Tugger_Project_V2/autonomous_tugger_ws
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH
source /opt/ros/lyrical/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export GZ_IP=127.0.0.1
export QT_QPA_PLATFORM=xcb
export CYCLONEDDS_URI='<CycloneDDS><Domain><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
python3 tools/mission_view/mission_view.py
```

Add `--no-spectator` to the last line for the camera windows only. The QR
window needs T2 running. Keep the panel (T6) minimised, not closed, during a
mission: it runs the mission.

## T7 — RViz (optional)

```bash
cd ~/Autonomous_Tugger_Project_V2/autonomous_tugger_ws
source /opt/ros/lyrical/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
rviz2 --ros-args -p use_sim_time:=true
```

## T8 — Checks

After starting T5, wait about 30 s, then check Nav2 and the controllers:

```bash
cd ~/Autonomous_Tugger_Project_V2/autonomous_tugger_ws
source /opt/ros/lyrical/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
ros2 lifecycle get /map_server
ros2 lifecycle get /planner_server
ros2 lifecycle get /controller_server
ros2 lifecycle get /docking/controller_server
ros2 control list_controllers
```

The map server and all three Nav2 servers must be `active`, and all four
controllers `active`. If the map server is `unconfigured`, T4a was missed: run it
now; Nav2 picks the map up without a restart.
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
export CYCLONEDDS_URI='<CycloneDDS><Domain><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
ros2 run rqt_image_view rqt_image_view
```

Select `/downward_camera/image_raw`, then `/qr_docking/debug_image`.

## T10 — Test obstacles (optional)

Run these while a simulation is up. Positions suit the mission HOME to Station 3.

```bash
cd ~/Autonomous_Tugger_Project_V2/autonomous_tugger_ws
source /opt/ros/lyrical/setup.bash
source install/setup.bash
export GZ_IP=127.0.0.1
python3 tools/mission_view/scene_objects.py block   # box in the side lane: robot takes another aisle
python3 tools/mission_view/scene_objects.py box     # box in the middle aisle at x = 12: robot goes around
python3 tools/mission_view/scene_objects.py cross   # a person crosses at x = 7 and leaves
python3 tools/mission_view/scene_objects.py pace    # a person keeps crossing; Ctrl+C to stop
python3 tools/mission_view/scene_objects.py clear   # remove them all
```

## Required Nav2 controller check

T5 must provide exactly these two FollowPath action servers (run it in the
T8 terminal, which has the Cyclone DDS settings):

```bash
ros2 action list | grep -E '^/follow_path$|^/docking/follow_path$'
```

Normal navigation uses `/follow_path` and `/tugger/nav_cmd_vel`.
Precision docking uses `/docking/follow_path` and `/tugger/dock_nav_cmd_vel`.
Both command streams pass through `tugger_sim_motion_gate` before reaching
the 4WS controller.

Speed profile:

- Straight route: up to **1.0 m/s** (0.95 m/s loaded).
- Curves: **0.6 m/s**. The route follower slows down before each curve,
  planned from the path; the motion gate also caps genuine curves at 0.6 m/s
  while preserving the requested curvature.
- Station approach and departure: **0.3 m/s**. The gate slows the robot further
  near stand legs using its stopping-distance check.
- Final camera correction: capped at **0.05 m/s**, loaded or not.
- Warning field occupied: **0.3 m/s**. Protective field occupied: stop.

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

T2 starts every robot node, including `qr_docking_vision`,
`docking_precision_controller`, `safety_monitor`, `obstacle_monitor` and
`tugger_sim_motion_gate`. Do not start these separately. The docking controller
accepts only `/tugger/docking/cmd_vel`, verifies fresh docking ownership, and
forwards to `/tugger/precision_cmd_vel`.

Do not start a second ground-truth bridge, map server, Nav2 launch, or operator panel.
