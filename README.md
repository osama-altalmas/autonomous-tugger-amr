# Autonomous Tugger AMR

Simulation of an autonomous four-wheel-steered (4WS) warehouse tugger, built on
ROS 2 Lyrical, Gazebo Sim 10 and Nav2. It drives generated routes between
stations, docks under cargo on floor markers to a 10 mm / 1° target, lifts and
delivers the cargo, and reacts to unexpected obstacles. No physical robot exists yet.

> **For AI assistants and new readers:** this file is written so you can install,
> run and explain the project without reading the source. Use
> [Where to change things](#where-to-change-things) to go straight to the right
> file. [TERMINAL_COMMANDS.md](TERMINAL_COMMANDS.md) has the full per-terminal
> startup with checks.

## Status

| Area | State |
|---|---|
| Route driving, docking, lift, cargo delivery, return to HOME | Working in simulation. About 0.5-0.7 cm rms from the path, under 2 cm at worst; docking within 10 mm |
| Laser localization: AMCL refined by a scan-to-map match, floor-marker correction at stations | Working in simulation, about 0.5 cm and 0.07° rms |
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
git clone https://github.com/osama-altalmas/autonomous-tugger-amr.git autonomous_tugger_ws
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
export CYCLONEDDS_URI='<CycloneDDS><Domain><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
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

The route follower's control law has its own C++ tests (after a build):

```bash
./build/tugger_navigation/test_path_tracking_law
```

## The robot and the warehouse

| Item | Value |
|---|---|
| Body | 0.90 × 0.60 m, wheelbase 0.58 m, track 0.44 m, wheel radius 0.055 m |
| Cargo plate | 1.3 × 1.3 m, lifted about 10 cm |
| Drive | Four wheels, each steered and driven |
| Motion modes | Car-like (counter-phase 4WS with Ackermann angles, tightest radius 0.51 m), turn in place (54°), crab (camera stage only) |
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

**Route following:** `tugger_navigation::CurvatureTrackingController`, a Nav2
controller plugin. Regulated Pure Pursuit still sets speed, direction, collision
stops and the end-of-path slow-down. Steering is the path's own curvature plus a
correction for sideways and heading error: gentle on straights (like a 1.2 m
look-ahead), firm near and in curves (like 0.6 m). The robot slows to the 0.6 m/s
curve speed before a curve, planned from the path. Pure pursuit alone cut route
corners by 10-17 cm. Route corners are exact circular arcs, so the steering holds
one angle through each corner.

**Steering:** counter-phase Ackermann. Front and rear axles steer opposite ways and
every wheel points around one turning centre, so no wheel slides sideways in a
curve. The inner wheel's 45° limit sets the tightest turn (0.51 m radius);
`src/tugger_control/tugger_control/steering_geometry.py` holds that limit for
every module, and a test checks `nav2.yaml` and the C++ defaults against it.

**Localization:** wheel odometry + IMU (EKF) → AMCL on the merged scan → sharpened by
a scan-to-map match (`refined_localization`) → corrected by floor markers at stations. Ground truth from Gazebo is used only to score tests;
the robot never reads it.
- `scan_merger` bins the two scanners into 0.5° slices published at each slice's
  centre (publishing the slice edge turned the scan and biased AMCL's heading by 0.3°).
- AMCL expects little odometry noise from straight driving and more from turning
  (`amcl.yaml`), so its estimate stays steady on straights but still absorbs wheel
  slip in curves.
- Near a station AMCL is paused and the pose rides on odometry from the exact
  marker fix. When the robot leaves, `marker_localization` re-seeds AMCL at the
  current pose before resuming, so the whole departure is not applied as one step
  (that jumped the pose by up to 37 cm).
- `refined_localization` publishes `map -> odom` (AMCL's `tf_broadcast` is off).
  After every AMCL update it fits both raw scans to the localization map, starting
  from AMCL's pose. A good fit (enough points on walls, small residual, within
  15 cm and 3° of AMCL) replaces AMCL's pose; otherwise AMCL's is used. Directions
  the scan cannot pin down (a featureless corridor) keep AMCL's value. It pauses in
  station zones. Its decisions are on `/tugger/localization/refinement`.
- Result in simulation: AMCL alone 2-3 cm; refined 0.5 cm rms and 0.07° heading.
- RViz view of the route, the true and estimated tracks and every correction:
  `tools/mission_view/tracking_view.sh` (T7 in TERMINAL_COMMANDS.md).

**Safety (`safety_monitor`, independent of mission logic):** at 1 m/s the protective
field reaches 0.84 m ahead and the warning field 2.34 m. Protective stops, warning
slows to 0.3 m/s. E-stops and bumper latch until **Reset safety**. A lost or stale sensor stops the robot.

**Obstacles (`obstacle_monitor` + `obstacle_policy`, separate from safety):** objects
not on the map are tracked up to 6 m ahead and classed as stationary or moving.

| Situation | Reaction |
|---|---|
| Object does not interfere with the path | Continue |
| Stationary object in the way, room to pass | Wait 5 s, then detour at normal route speed with a 0.5 m gap (from the cargo edge when loaded), rejoin the route |
| No room to pass | Take another aisle through the route network |
| Moving object | Stop and wait; after 30 s look for another path |
| Interfering object at a station | Hold and alert until it is gone |

## Repository map

| Path | Purpose |
|---|---|
| `src/tugger_description` | Robot model (URDF/Xacro, meshes) |
| `src/tugger_gazebo` | Warehouse world `tugger_warehouse.sdf`, simulation launch, controllers, QR assets, C++ Gazebo plugins |
| `src/tugger_control` | Python nodes, operator panel, all mission logic, tests |
| `src/tugger_navigation` | Nav2 and AMCL configuration, maps, custom RRT* Reeds-Shepp planner plugin, route-following controller plugin |
| `src/tugger_ground_truth` | True pose from Gazebo, for evaluation only |
| `tools/v1_manhattan_bezier` | Route generator and the generated routes (`mission_paths/`, `routes/`, `phase14/`) |
| `tools/mission_approaches` | Generated station gate approaches |
| `tools/warehouse_validation` | Builds the planning and localization maps from the world file |
| `tools/mission_validation` | Headless stack, mission runner, repeat-run campaign |
| `tools/mission_view` | Camera windows, follow view, RViz tracking view, test obstacle placement |
| `tools/rrt_star_tests` | C++ tests for the planner |

Nodes in `src/tugger_control/tugger_control/` (all started by `control.launch.py`):
`four_ws_controller`, `four_ws_odometry`, `sim_motion_gate`, `safety_monitor`,
`scan_merger`, `obstacle_monitor`, `marker_localization`, `qr_docking_vision`,
`docking_precision_controller`, `cargo_estimator`, `cargo_id_reader`.
`refined_localization` is started with AMCL by `tugger_navigation/localization.launch.py`. The other
files there are libraries. `operator_panel.py` is the mission engine and rqt panel.

`nav2_v1_static_planning.launch.py`, `nav2_v1_static.yaml`, `slam.launch.py` and the
`BidirectionalPlanner` plugin are older or optional and are not in the normal startup.

## Where to change things

| To change | Edit |
|---|---|
| Route speed, look-ahead, arrival tolerance | `src/tugger_navigation/config/nav2.yaml` |
| Route follower steering and curve speed (`tracking_*`, `curve_speed*`) | `FollowPath` in `src/tugger_navigation/config/nav2.yaml`; law in `src/tugger_navigation/include/tugger_navigation/path_tracking_law.hpp` |
| Localization noise model | `src/tugger_navigation/config/amcl.yaml` |
| Scan-match acceptance limits | `src/tugger_control/tugger_control/scan_refinement.py` |
| Turning limit (wheel angle, tightest radius) | `src/tugger_control/tugger_control/steering_geometry.py` |
| Station zones, marker corrections, re-seed on departure | `src/tugger_control/tugger_control/marker_localization.py` |
| Route shape (grid search, corner rounding, staging distance) | `tools/v1_manhattan_bezier/build_route_database_v1.py`, `plan_routes.py`, then `tools/regenerate_paths_from_world.sh` |
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

- Routes are generated, never hand-written. Station centres come from the world
  file; each gate's staging pose is 2.0 m out along the station axis, and routes
  end exactly on it. Regenerated routes are not approved and need review before use.
- The station config (`src/tugger_control/config/operator_stations.json`) is written
  by the generator (`sync_station_entries.py`); do not edit its poses by hand.
- A station is left by the gate the robot entered, unless turn in place is on and faster.
- Nothing commands the wheels except through `sim_motion_gate`.
- Do not start a second copy of any launch file; duplicate nodes break the gate's checks.
- All terminals must use `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` and the `CYCLONEDDS_URI` setting from TERMINAL_COMMANDS.md.

## Common problems

| Symptom | Cause and fix |
|---|---|
| `action server not available`, Nav2 stuck | Leftover processes. Run Step 0 in TERMINAL_COMMANDS.md and restart |
| Robot has no `map` position | AMCL (T4b) not running, or started before its map. Follow the order in TERMINAL_COMMANDS.md |
| "Path validation failed or reported an obstacle; RUN is not enabled" with nothing in the way | Map server not activated, so Nav2's costmap has no map. Run T4a; no restart needed |
| `Failed to find a free participant index for domain 0` | The running stack used Cyclone DDS's default of about 10 processes. Every terminal block in TERMINAL_COMMANDS.md sets `CYCLONEDDS_URI` to raise it; add that line to any other terminal |
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
- AMCL alone sits 2-3 cm off, with a map-dependent offset of about 2 cm. The scan
  match removes it, but whenever a fit is refused the robot uses AMCL's pose for
  that update.
- Rare camera-stage holds: one HOME dock held 200 s on a stale QR heading (cause
  not yet known; the hold message now names it).
- The cargo height-clearance guard reads Gazebo's exact geometry; a real robot
  needs that from its own sensors.
- Motor overload and per-wheel load are not modelled.

## Planned

1. A better localization fallback: remove AMCL's own offset and keep AMCL close
   to the refined pose, so a refused fit costs little.
2. Fault testing: reflections, glass, worn markers, moved pallets, sensor dropouts.
3. Then a new branch for the differential-drive variant and AI (YOLO perception,
   reinforcement learning as a small, bounded correction on the path follower).
4. A headless mission node before moving to hardware.

## License

No license has been chosen yet.
