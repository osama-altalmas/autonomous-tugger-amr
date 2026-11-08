# Autonomous Tugger AMR

Simulation of an autonomous four-wheel-steered (4WS) warehouse tugger, built on
ROS 2 Lyrical, Gazebo Sim 10 and Nav2. It drives generated routes between
stations, docks under cargo on floor markers to a 10 mm / 1° target, lifts and
delivers the cargo, and reacts to unexpected obstacles. No physical robot exists yet.

> **For AI assistants and new readers:** this file is written so you can install,
> run and explain the project without reading the source. Use
> [Where to change things](#where-to-change-things) to go straight to the right
> file. [TERMINAL_COMMANDS.md](TERMINAL_COMMANDS.md) has the full per-terminal
> startup with checks. Work on the `nav2-development` branch; `main` is older.

## Status

| Area | State |
|---|---|
| Route driving, docking, lift, cargo delivery, return to HOME | Working in simulation |
| Laser localization (AMCL) with floor-marker correction | Working in simulation |
| Safety: protective and warning fields, E-stops, bumper, stale-sensor stop | Working in simulation |
| Turn in place (optional checkbox) and crab motion at the docking camera stage | Working in simulation |
| Obstacle handling | First version. Static objects, blocked aisles and station blocks tested at fixed positions. A person pacing for a long time is not proven end to end |
| AI (YOLO perception, reinforcement learning) | Planned, nothing implemented |
| Real hardware | None. Safety values follow ISO 3691-4 practice but are not certified |

## Requirements

Tested on Ubuntu 26.04 with ROS 2 Lyrical and Gazebo Sim 10 (8 cores, 14 GB RAM).
The simulation runs slower than real time on that laptop.

1. Install ROS 2 Lyrical from the [official instructions](https://docs.ros.org/en/lyrical/Installation.html).
2. Install the packages this project needs:

```bash
sudo apt update
sudo apt install \
  ros-lyrical-desktop ros-lyrical-navigation2 ros-lyrical-nav2-bringup \
  ros-lyrical-ros-gz ros-lyrical-gz-ros2-control ros-lyrical-ros2-control \
  ros-lyrical-ros2-controllers ros-lyrical-robot-localization \
  ros-lyrical-rmw-cyclonedds-cpp ros-lyrical-slam-toolbox ros-lyrical-xacro \
  ros-lyrical-rqt-gui-py ros-lyrical-rqt-image-view ros-lyrical-tf2-tools \
  ros-lyrical-cv-bridge ros-lyrical-ompl ros-lyrical-sensor-msgs-py \
  ros-lyrical-joint-state-publisher-gui \
  python3-colcon-common-extensions python3-opencv python3-numpy python3-scipy \
  python3-yaml python3-matplotlib python3-pil python3-pytest nlohmann-json3-dev
```

There are no pip requirements. The Gazebo Python bindings come from the ROS
vendor packages and are importable only after sourcing ROS.

`rosdep` cannot resolve two keys in the package files (`nlohmann-json3-dev`,
`ros-lyrical-gz-msgs-vendor`), so use the list above.

## Build

```bash
git clone -b nav2-development https://github.com/osama-altalmas/autonomous-tugger-amr.git autonomous_tugger_ws
cd autonomous_tugger_ws
source /opt/ros/lyrical/setup.bash
colcon build --symlink-install
```

`--symlink-install` is **required**: the operator panel finds the path files in
`tools/` relative to its source file, and a normal install breaks that.

## Run

**With windows (Gazebo and operator panel):** follow
[TERMINAL_COMMANDS.md](TERMINAL_COMMANDS.md), one terminal per component. The
order matters. In the panel: tick **Arm**, press **Check / plan Station N**,
optionally **ATTACH** or **DETACH**, then **RUN selected mission**.

**Without windows (one command, for tests):** this starts everything on an
isolated ROS domain, runs a mission and writes a JSON result.

```bash
source /opt/ros/lyrical/setup.bash && source install/setup.bash
export GZ_IP=127.0.0.1 QT_QPA_PLATFORM=offscreen RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
tools/mission_validation/isolated_stack.sh up 77
until [ -f /tmp/tugger_validation/domain_77/nav2_ready ]; do sleep 2; done
ROS_DOMAIN_ID=77 GZ_PARTITION=tugger_validation_77 python3 tools/mission_validation/mission_runner.py \
  cargo_station_0 cargo_dropoff_northwest HOME --output /tmp/result.json --timeout 900
tools/mission_validation/isolated_stack.sh down 77
```

Each destination may end in `:attach`, `:detach` or `:none`. Logs are in
`/tmp/tugger_validation/domain_77/`.

**Automated tests:**

```bash
source /opt/ros/lyrical/setup.bash && source install/setup.bash
python3 -m pytest -q src/tugger_control/test
```

`test_flake8`, `test_pep257` and `test_mypy` currently fail on code style only;
that cleanup has not been done. All other tests should pass.

## The robot and the warehouse

| Item | Value |
|---|---|
| Body | 0.90 × 0.60 m, wheelbase 0.58 m, track 0.44 m, wheel radius 0.055 m |
| Cargo plate | 1.3 × 1.3 m, lifted about 10 cm |
| Drive | Four wheels, each steered and driven |
| Motion modes | Car-like (counter-phase 4WS), turn in place (54°), crab (camera stage only) |
| Laser scanners | Two, at (0.41, 0.26) and (−0.41, −0.26) m, 275° each, merged into `/scan` |
| Depth cameras | Front and rear |
| Other sensors | IMU, wheel odometry, downward QR camera, top cargo QR camera, bumper |

Sensors have simulated noise only; faults such as glass or worn markers are not modelled.

| Name in panel | Name in code | Position (x, y) |
|---|---|---|
| HOME | `HOME` | (0, 0) |
| Station 1 | `cargo_station_0` | (−17.5, −7) |
| Station 2 | `cargo_dropoff_northwest` | (−17.5, 7) |
| Station 3 | `cargo_station_3` | (17.5, 7) |
| Station 4 | `cargo_dropoff_southeast` | (17.5, −7) |

| Speed limit | m/s |
|---|---|
| Straight route | 1.0 |
| Curves | 0.6 |
| Loaded route request | 0.95 |
| Station approach and departure | 0.3 |
| Warning field active | 0.3 |
| Camera stage | 0.05 |

## How it works

**A mission:** drive a generated single-segment route → enter the station gate
at 0.3 m/s → final alignment on the floor QR marker with the downward camera →
lift or lower the cargo → leave by the same gate → continue to the next stop.

**All motion passes through one gate, which fails closed:**

```
Nav2 route controller    → /tugger/nav_cmd_vel      ┐
Nav2 docking controller  → /tugger/dock_nav_cmd_vel ├→ sim_motion_gate → /tugger/guarded_cmd_vel → four_ws_controller → wheels
Camera docking stage     → /tugger/precision_cmd_vel│   (one owner at a time: /tugger/motion_owner;
Turn in place            → /tugger/rotate_cmd_vel   ┘    stops on /tugger/safety/state)
```

**Localization:** wheel odometry + IMU (EKF) → AMCL on the merged scan → corrected
by floor markers at stations. Ground truth from Gazebo is used only to score tests;
the robot never reads it.

**Safety (`safety_monitor`, independent of mission logic):** at 1 m/s the protective
field reaches 0.84 m ahead and the warning field 2.34 m. Protective stops, warning
slows to 0.3 m/s. E-stops and bumper latch until **Reset safety**. A lost or stale sensor stops the robot.

**Obstacles (`obstacle_monitor` + `obstacle_policy`, separate from safety):** objects
not on the map are tracked up to 6 m ahead and classed as stationary or moving.

| Situation | Reaction |
|---|---|
| Object does not interfere with the path | Continue |
| Stationary object in the way, room to pass | Wait 10 s, then detour at 0.3 m/s with a 0.5 m gap (from the cargo edge when loaded), rejoin the route |
| No room to pass | Take another aisle through the route network |
| Moving object | Stop and wait; after 30 s look for another path |
| Interfering object at a station | Hold and alert until it is gone |

## Repository map

| Path | Purpose |
|---|---|
| `src/tugger_description` | Robot model (URDF/Xacro, meshes) |
| `src/tugger_gazebo` | Warehouse world `tugger_warehouse.sdf`, simulation launch, controllers, QR assets, C++ Gazebo plugins |
| `src/tugger_control` | Python nodes, operator panel, all mission logic, tests |
| `src/tugger_navigation` | Nav2 and AMCL configuration, maps, custom RRT* Reeds-Shepp planner plugin |
| `src/tugger_ground_truth` | True pose from Gazebo, for evaluation only |
| `tools/v1_manhattan_bezier` | Route generator and the generated routes (`mission_paths/`, `routes/`, `phase14/`) |
| `tools/mission_approaches` | Generated station gate approaches |
| `tools/warehouse_validation` | Builds the planning and localization maps from the world file |
| `tools/mission_validation` | Headless stack, mission runner, repeat-run campaign |
| `tools/mission_view` | Camera windows, follow view, test obstacle placement |
| `tools/rrt_star_tests` | C++ tests for the planner |

Nodes in `src/tugger_control/tugger_control/` (all started by `control.launch.py`):
`four_ws_controller`, `four_ws_odometry`, `sim_motion_gate`, `safety_monitor`,
`scan_merger`, `obstacle_monitor`, `marker_localization`, `qr_docking_vision`,
`docking_precision_controller`, `cargo_estimator`, `cargo_id_reader`. The other
files there are libraries. `operator_panel.py` is the mission engine and rqt panel.

`nav2_v1_static_planning.launch.py`, `nav2_v1_static.yaml`, `slam.launch.py` and the
`BidirectionalPlanner` plugin are older or optional and are not in the normal startup.

## Where to change things

| To change | Edit |
|---|---|
| Route speed, look-ahead, arrival tolerance | `src/tugger_navigation/config/nav2.yaml` |
| Robot speed, acceleration limits | `src/tugger_control/config/sim_speed.yaml` |
| Limits enforced on every command | `src/tugger_control/tugger_control/sim_motion_gate.py` |
| Safety field sizes | `src/tugger_control/tugger_control/safety_fields.py` |
| Obstacle gaps, wait times, reactions | `src/tugger_control/tugger_control/obstacle_policy.py` |
| Obstacle detection | `src/tugger_control/tugger_control/obstacle_monitor.py` |
| Robot geometry and sensor placement | `src/tugger_description/urdf/tugger.urdf.xacro` |
| Warehouse or stations | `src/tugger_gazebo/worlds/tugger_warehouse.sdf`, then run `tools/regenerate_paths_from_world.sh` |
| Test obstacles | `tools/mission_view/scene_objects.py` (`block`, `box`, `cross`, `pace`, `clear`) |

After a config or C++ change, rebuild and restart the affected terminals.

## Rules the code relies on

- Routes are generated, never hand-written. Regenerated routes need review before use.
- A station is left by the gate the robot entered, unless turn in place is on and faster.
- Nothing commands the wheels except through `sim_motion_gate`.
- Do not start a second copy of any launch file; duplicate nodes break the gate's checks.
- All terminals must use `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`.

## Common problems

| Symptom | Cause and fix |
|---|---|
| `action server not available`, Nav2 stuck | Leftover processes. Run Step 0 in TERMINAL_COMMANDS.md and restart |
| Robot has no `map` position | Map server not activated, or AMCL started before it. Follow the order in TERMINAL_COMMANDS.md |
| Panel cannot find routes | Built without `--symlink-install` |
| Robot stopped and will not restart | Safety latched. Clear the cause and press **Reset safety** |
| Topics invisible between terminals | One terminal is missing the Cyclone DDS setting |
| GUI apps crash when started from the VS Code Snap terminal | Snap library conflict. Use a normal terminal |

## Known limitations

- Simulation only.
- Obstacle logic does not see objects sitting above the scanner plane.
- Test obstacles are at fixed positions.
- Obstacle positions are not remembered, so a replan avoids only where an object is now.
- A bumper hit needs a manual reset, and there is no reverse recovery.
- Path following can weave from side to side; it has not been measured and tuned yet.
- Motor overload and per-wheel load are not modelled.

## Planned

YOLO perception to classify objects, reinforcement learning as a small, bounded
correction on top of the path follower, then a headless mission node before moving to hardware.

## License

No license has been chosen yet.
