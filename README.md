# Autonomous Tugger AMR

Simulation of an autonomous four-wheel-steered (4WS) warehouse tugger, built on
ROS 2 Lyrical, Gazebo Sim 10 and Nav2. It drives generated routes between
stations, docks under cargo on floor markers to a 10 mm / 1° target, lifts and
delivers the cargo, and reacts to unexpected obstacles. It has been tested against
realistic warehouse faults (sensors, markers, moved racks, people, worn tyres).

**This branch, AMR-V3,** is the finished 4WS version (branch `main`) plus AI. Done so
far: a learned floor-marker reader (YOLO11n-pose) that also reads worn and dirty
markers and docks with them. Next: obstacle recognition, then reinforcement learning.
No physical robot exists yet.

> **For AI assistants and new readers:** this file is written so you can install,
> run and explain the project without reading the source. Use
> [Where to change things](#where-to-change-things) to go straight to the right
> file. [TERMINAL_COMMANDS.md](TERMINAL_COMMANDS.md) has the full per-terminal
> startup with checks.

## Status

| Area | State |
|---|---|
| Route driving, docking, lift, cargo delivery, return to HOME | Working in simulation. About 0.5-0.7 cm rms from the path, under 2 cm at worst; docking within 10 mm |
| Laser localization: AMCL refined by a scan-to-map match, floor-marker correction at stations | Working in simulation, about 0.5 cm and 0.07° rms; AMCL kept near the refined pose as a backup (about 2 cm when used alone) |
| Localization checks: integrity check with relocalization, lost state, odometry scale calibration | Working in simulation |
| Safety: protective and warning fields, E-stops, bumper, stale-sensor stop, blind-scanner check | Working in simulation |
| Turn in place (optional checkbox) and crab motion at the docking camera stage | Working in simulation |
| Obstacle handling | Working in simulation: standing objects, blocked aisles, station blocks, people walking across the route |
| Fault testing (simulation) | Done (2026-11-14): no collision, no wrong dock and no silent stop in any scenario. See [Fault testing](#fault-testing) |
| AI floor-marker reader (YOLO11n-pose + tag decoding + red-frame fit), LED ring light | Working in simulation and used for docking: 82 % of damaged test markers read (classic reader 59 %), never a wrong station, docks 1.2-6.6 mm. See [Floor-marker reader (AI)](#floor-marker-reader-ai) |
| AI obstacle recognition, reinforcement learning | Next on this branch |
| Differential drive | Planned on its own branch, made from this one |
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

There are no pip requirements to run the robot: the marker model runs on the CPU
with OpenCV's DNN module. Training needs a GPU and Ultralytics 8.3.40; it runs on
Google Colab with the notebooks in `tools/ai/marker_dataset/`. The Gazebo Python
bindings come from the ROS vendor packages and are importable only after sourcing ROS.

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
| Other sensors | IMU, wheel odometry, downward marker camera with a 12-LED ring light, top cargo QR camera, bumper |

Sensors have simulated noise. For tests, realistic faults can be switched on
(dirty or covered scanner windows, reflections, IMU drift, worn tyres, worn
markers); see [Fault testing](#fault-testing).

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
at 0.3 m/s → final alignment on the floor marker with the downward camera →
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
- A fit far from AMCL is still used when it continues the refined pose of a moment
  ago (AMCL drifted, the robot did not jump). In an aisle, where the scan pins only
  the cross-aisle position and heading, only those are corrected.
- Backup: AMCL is the fallback whenever a fit is refused. After 10 good fits in a
  row it is re-seeded at the refined pose when it is more than 3 cm or 0.5° away (at
  most every 5 s, never at a station), and AMCL reads a half-cell-shifted copy of the
  map that removes its own offset. If AMCL and the scan match disagree by more than
  10 cm or 3° for 3 s, the panel shows a warning.
- Odometry calibration: the distance between good fits, measured by the scan match,
  corrects the wheel odometry's scale (worn or compressed tyres; a 2% error put the
  robot 5.5 cm off at a camera stage). Limits 0.95-1.05; a correction over 3% is reported.
- Integrity check, once a second, also at stations: the current scan is scored
  against the map. Three poor scores in a row start a search around the pose; a
  clearly better pose is taken over (relocalized, e.g. after the robot was pushed by
  hand). If none is found the robot is **lost**: motion stops and RUN is refused until
  the pose is known again.
- Result in simulation: AMCL alone 2-3 cm; refined 0.5 cm rms and 0.07° heading;
  with every fit refused (AMCL backup only) about 2 cm, tracking 1.2-1.3 cm, docks 5-7 mm.
- RViz view of the route, the true and estimated tracks and every correction:
  `tools/mission_view/tracking_view.sh` (T7 in TERMINAL_COMMANDS.md).

**Safety (`safety_monitor`, independent of mission logic):** at 1 m/s the protective
field reaches 0.84 m ahead and the warning field 2.34 m. Protective stops, warning
slows to 0.3 m/s. E-stops and bumper latch until **Reset safety**. A lost or stale sensor stops the robot.
A scanner whose window is covered or blinded (no return over 60° that stays while the
robot drives 1.5 m) also stops it, with a message naming the scanner.

**Obstacles (`obstacle_monitor` + `obstacle_policy`, separate from safety):** objects
not on the map are tracked up to 6 m ahead, judged by their outline (not a circle
around them) and classed as stationary or moving. Gaps are measured from the robot's
side, or from the cargo's edge when loaded.

| Situation | Reaction |
|---|---|
| Object does not interfere with the path | Continue |
| Standing object 15-50 cm from the path's edge | Pass slowly at 0.3 m/s |
| Standing object closer than 15 cm (in the way), room to pass | Wait 5 s, then detour at normal route speed with a 0.5 m gap, rejoin the route |
| No room to pass | Take another aisle through the route network |
| Neither a detour nor another aisle | Wait on the path until it clears; a re-plan that finds nothing is reported as **MISSION STOPPED** |
| Moving object (a person) | Keep 50 cm; stop and wait; after 30 s look for another path |
| Interfering object at a station | Hold and alert until it is gone |

**Floor markers:** each station has a 12 cm red frame around a white plate with four
AprilTags (36h11), one per quarter, all naming the station. A marker that cannot be
read (worn, covered) or a red frame that cannot be measured is refused with a panel
message naming it; the robot does not dock blind, and backs out of the station on the
next mission.

### Floor-marker reader (AI)

The downward camera sits about 6 cm above the floor, under the robot and often under
cargo, where no room light reaches. A ring of 12 LEDs around the lens lights the
marker the same way at every station (a real robot also needs a brush skirt to keep
room light out; it is not modelled).

`qr_docking_vision` reads every image in three steps:

1. **Find the marker:** a YOLO11n-pose model (`src/tugger_control/models/marker_reader_v3.onnx`,
   about 130 ms per image on the laptop CPU) returns the marker, a confidence and 20
   points: the frame's 4 corners and the 4 corners of each tag, numbered from the
   image's up-left corner (a square marker looks the same turned by 90°, so the model
   is asked only what it can see).
2. **Name the station:** the tags are read through those points; the turn at which
   each tag sits in its own quarter gives the marker's own orientation. The station
   is never learnt, so a new station is one line in `STATION_TAG_BASE`
   (`docking_markers.py`), not a retraining.
3. **Measure the frame:** the known square is fitted to the visible edge of the red
   frame (`marker_geometry.py`); two adjacent sides are enough. Fitted frames were
   within 1.5 px (0.7 mm) of the true frame, median 0.3 px.

The classic OpenCV tag reader runs beside it as a check: if both name a station and
differ, nothing is published and the robot holds. The model runs only when enough of
the red frame is in view. The state is published on `/tugger/marker_reader`:
**OK**; **ALERT** (confidence below 80 %, or only one tag left: clean or replace the
marker soon); **FAULT** (below 50 %, no tag, wrong station, frame not measurable, or
the LED light too dark). Set `marker_reader:=classic` on `qr_docking_vision` to dock
with the classic reader only.

| On 1,514 unseen test images | Classic tags | AI v2 | AI v3 (used) |
|---|---|---|---|
| Right station, 401 real markers | 59 % | 47 % | **82 %** |
| Wrong station | 0 | 11 | **0** |
| Heavily damaged markers read (94) | 9 | 12 | **42** |
| Empty floors taken for a marker (50) | 0 | 0 | **0** |

Training data is made in Gazebo (one clean image per robot pose, labelled
automatically from the tags) and damaged afterwards by augmentation: dust, scratches,
tape, smears, wiped frame sides, weak or failed LEDs. Training runs on Google Colab
(`tools/ai/marker_dataset/train_markers_v3_colab.ipynb`).
The full record of every iteration, mistake and result:
[ai_data/PROCESS.md](ai_data/PROCESS.md).

## Fault testing

Done in simulation on 2026-11-14 (`tools/fault_testing/`), with faults chosen from
what real warehouse robots run into. Pass meant: the robot either keeps working
within limits or stops safely with a clear panel message; never a collision, a wrong
dock or a silent stop. Every scenario passed after eight fixes to the robot code.

| Fault | Result (after the fixes) |
|---|---|
| Cart standing beside the path | Passed: objects are judged by their outline, not a circle around them |
| Rack moved 20 cm (stale map) | Localization within 2 cm (AMCL re-centred in aisles too), docked at 5.2 mm; before the fix it drifted 18 cm |
| Person standing in the aisle; people crossing or stepping out | Waits on the path and continues by itself; no change was needed for crossing people |
| No route left after a re-plan | Panel reports **MISSION STOPPED** with the reason (before: silent) |
| Film on a scanner window (120° blind) | Protective stop, "window dirty or covered?" |
| Heavy reflections, gyro drift 15°/h, slippery floor patch | Handled with no change |
| Wheels 2% smaller (worn tyres) | Odometry scale measured as 0.981 (true 0.980), HOME docked at 5.7 mm; before: 5.5 cm off |
| Worn floor QR | Panel says the QR cannot be read and should be cleaned or replaced; the robot backs out of the station. (On this branch the AI reader reads most worn markers; see [Floor-marker reader (AI)](#floor-marker-reader-ai)) |
| Stations nudged 3-4 cm | Docked within 5.9 mm |
| Robot pushed 67 cm and 80 cm by hand | Found its true position within about 5 s; if nothing fits clearly, **LOCALIZATION LOST** and RUN is refused |
| Scanner goes silent; localization program crashes | Protective stop and restart; panel says **LOCALIZATION STOPPED** |

Scenario files: `tools/fault_testing/schedules/`; how to run one:
`tools/fault_testing/README.md`. The simulation launch takes `sensor_faults:=true`
(scanners and IMU pass through the fault relay) and `sim_wheel_radius:=` (worn tyres);
without them the simulation is unchanged.

## Repository map

| Path | Purpose |
|---|---|
| `src/tugger_description` | Robot model (URDF/Xacro, meshes) |
| `src/tugger_gazebo` | Warehouse world `tugger_warehouse.sdf`, simulation launch, controllers, marker textures (`media/markers/`), cargo tags, C++ Gazebo plugins |
| `src/tugger_control` | Python nodes, operator panel, all mission logic, tests; `models/` holds the trained marker model |
| `src/tugger_navigation` | Nav2 and AMCL configuration, maps, custom RRT* Reeds-Shepp planner plugin, route-following controller plugin |
| `src/tugger_ground_truth` | True pose from Gazebo, for evaluation only |
| `tools/v1_manhattan_bezier` | Route generator and the generated routes (`mission_paths/`, `routes/`, `phase14/`) |
| `tools/mission_approaches` | Generated station gate approaches |
| `tools/warehouse_validation` | Builds the planning and localization maps from the world file |
| `tools/mission_validation` | Headless stack, mission runner, repeat-run campaign |
| `tools/mission_view` | Camera windows, follow view, RViz tracking view, test obstacle placement |
| `tools/fault_testing` | Fault relay, timed fault scenarios, the campaign's scenario files |
| `tools/rrt_star_tests` | C++ tests for the planner |
| `tools/ai/marker_dataset` | Marker dataset generator (Gazebo), damage augmentation, Colab training notebooks, reader comparison |
| `ai_data` | AI record `PROCESS.md` with its charts and result summaries (datasets and models stay local, several GB) |

Nodes in `src/tugger_control/tugger_control/` (all started by `control.launch.py`):
`four_ws_controller`, `four_ws_odometry`, `sim_motion_gate`, `safety_monitor`,
`scan_merger`, `obstacle_monitor`, `marker_localization`, `qr_docking_vision`,
`docking_precision_controller`, `cargo_estimator`, `cargo_id_reader`.
`refined_localization` is started with AMCL by `tugger_navigation/localization.launch.py`. The other
files there are libraries. `operator_panel.py` is the mission engine and rqt panel.

`slam.launch.py` (mapping a real warehouse) and the `BidirectionalPlanner` plugin are
optional and are not in the normal startup.

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
| Slow-pass gap and speed, moving-object gap | `GAP_ROUTE_MIN_M`, `SLOW_PASS_SPEED_M_S`, `GAP_ROUTE_M` in `obstacle_policy.py` |
| AMCL re-centre, integrity check, odometry calibration | Constants at the top of `src/tugger_control/tugger_control/refined_localization.py` |
| Blind-scanner check | `BLIND_SECTOR_DEG`, `BLIND_TRAVEL_M` in `safety_monitor.py` |
| Fault scenarios | `tools/fault_testing/schedules/*.json` |
| Marker reader: ALERT / FAULT confidence, light limits, camera height | Constants at the top of `src/tugger_control/tugger_control/marker_reader.py` |
| Frame fit tolerances | `src/tugger_control/tugger_control/marker_geometry.py` |
| Stations and their tag IDs | `STATION_TAG_BASE` in `src/tugger_control/tugger_control/docking_markers.py` |
| Docking reader (AI or classic), red-pixel gate | `marker_reader` parameter, `MIN_RED_PIXELS` in `qr_docking_vision.py` |
| LED ring (number, radius, brightness) | `led_*` properties in `src/tugger_description/urdf/tugger.urdf.xacro` (changing the URDF needs `tools/regenerate_paths_from_world.sh`: the routes check the robot file) |
| Retrain the marker model | `tools/ai/marker_dataset/` (generate, `augment.py`, Colab notebook), then replace `src/tugger_control/models/marker_reader_v3.onnx` |

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
- Obstacle positions are not remembered, so a replan avoids only where an object is now.
- A bumper hit needs a manual reset, and there is no reverse recovery.
- The slow-pass speed (0.3 m/s) is an assumption, not a standard value.
- The AI reader is trained on simulated images only; real camera photos are needed
  before it can be trusted on a real robot. Heavily damaged markers (about half of
  the heaviest test damage) are still refused: the robot does not dock there until
  the marker is cleaned.
- In the simulated marker the tag plate lies 1 mm above the red frame; the reader
  takes this into account (camera height measured to the frame, 5.63 cm).
- Rarely, after setting cargo down at Station 4, the simulated robot stays held in
  place while its wheels turn (seen once in four runs); not yet investigated.
- After a fault that leaves the robot off its route, it does not plan its own way
  back; the operator restarts the mission.
- AMCL alone sits about 2 cm off; it is used only for updates where the scan match
  is refused.
- The cargo height-clearance guard reads Gazebo's exact geometry; a real robot
  needs that from its own sensors.
- Motor overload and per-wheel load are not modelled.

## Planned

On this branch (AMR-V3), in order:

1. Done: learned floor-marker reader.
2. Obstacle recognition on the depth cameras (person, pallet, cart, forklift).
3. Reinforcement learning as a small, bounded correction (docking fine control,
   recovery after leaving the route); the safety layer stays deterministic.

Then a differential-drive variant on its own branch made from this one, and a
headless mission node before moving to hardware.

## License

GNU Affero General Public License v3.0 (AGPL-3.0-only); see [LICENSE](LICENSE). Every
package in `src/` declares the same. This branch uses Ultralytics YOLO (AGPL-3.0) to
train the marker model, so the branch and the model are AGPL-3.0. The `main` branch
(the 4WS version without AI) is Apache-2.0.
