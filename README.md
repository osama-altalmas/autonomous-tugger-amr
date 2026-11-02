# Autonomous Tugger AMR

An autonomous four-wheel-steering (4WS) mobile robot developed using ROS 2 Lyrical, Gazebo Sim 10, and Nav2.

The project demonstrates autonomous warehouse navigation, precision docking, cargo lifting, transportation, and unloading in a simulated industrial environment.

## Project Status

**Successfully demonstrated an end-to-end autonomous cargo transportation mission.**

The robot transported a movable stand carrying a box from the southwest pickup station to the northwest delivery station.

The demonstrated mission consisted of:

1. Autonomous navigation to the pickup approach.
2. Precision docking beneath the cargo assembly.
3. Raising the lifting platform.
4. Retreating from the pickup station.
5. Autonomous navigation while carrying cargo.
6. Final navigation to the delivery position.
7. Lowering the lift and releasing the cargo.

The mission was completed using separate execution stages.

## Demonstrated Results

| Metric | Result |
|---|---|
| Pickup station | Southwest (-17.5, -7.0) |
| Delivery station | Northwest (-17.5, 7.0) |
| Cargo | Movable stand and box |
| Navigation | Nav2 |
| Planner | Custom bidirectional planner |
| Controller | Regulated Pure Pursuit |
| Maximum configured navigation speed | 0.20 m/s |
| Final cargo positioning error | Approximately 1.82 cm |
| Final cargo orientation error | Approximately 1.63 degrees |
| Cargo release | Visually verified in Gazebo |

These measurements are from one successful simulation run.

## Technology Stack

- Ubuntu 26.04
- ROS 2 Lyrical
- Gazebo Sim 10
- Nav2
- Python and C++
- URDF/Xacro
- ROS 2 Control
- TF2

## Robot Architecture

The tugger uses a four-wheel-steering architecture with four steering joints and four wheel joints.

The project includes:

- Custom 4WS motion controller
- Wheel odometry
- Gazebo simulation
- Ground-truth localization bridge
- Nav2 path planning and control
- Custom bidirectional planner
- Cargo docking and lifting system
- Warehouse navigation environment

## Navigation Architecture

A key requirement is that each Nav2 navigation leg must remain entirely forward-only or reverse-only.

The custom planner generates candidate paths, and the mission validates direction consistency before execution.

Direction changes are permitted between separate mission phases, such as docking and retreat.

The navigation system uses:

- Custom tugger_navigation::BidirectionalPlanner
- Nav2 Regulated Pure Pursuit
- Wheel odometry
- Static warehouse map
- Local and global costmaps

## ROS 2 Packages

| Package | Description |
|---|---|
| tugger_description | Robot URDF/Xacro |
| tugger_gazebo | Gazebo simulation and warehouse |
| tugger_control | 4WS controller and odometry |
| tugger_navigation | Nav2 and custom planner |
| tugger_ground_truth | Simulation ground-truth integration |

## Build

From the workspace directory:

    source /opt/ros/lyrical/setup.bash
    colcon build --symlink-install
    source install/setup.bash

## Simulation and Navigation

Available launch files:

    src/tugger_gazebo/launch/simulation.launch.py
    src/tugger_control/launch/control.launch.py
    src/tugger_navigation/launch/nav2.launch.py
    src/tugger_navigation/launch/slam.launch.py

The successful mission used Gazebo, wheel odometry, ground-truth localization, a static map server, and Nav2.

A complete clean-install startup procedure is still to be documented and verified.

## Cargo Mission

Mission implementation:

    tools/cargo_mission/mission.py

Direction validation:

    tools/cargo_mission/path_direction_validator.py

The following examples assume the simulation, localization, map server, and Nav2 are running, and the current directory is the workspace root.

### Preview pickup route

    PYTHONPATH="$PWD/tools/warehouse_navigation:$PWD/tools/cargo_mission:$PYTHONPATH" python3 tools/cargo_mission/mission.py --pickup sw --dropoff nw --no-smoothing --preview

### Navigate to pickup approach

    PYTHONPATH="$PWD/tools/warehouse_navigation:$PWD/tools/cargo_mission:$PYTHONPATH" python3 tools/cargo_mission/mission.py --pickup sw --dropoff nw --no-smoothing

### Dock and lift

    PYTHONPATH="$PWD/tools/warehouse_navigation:$PWD/tools/cargo_mission:$PYTHONPATH" python3 tools/cargo_mission/mission.py --pickup sw --dropoff nw --no-smoothing --enable-docking --dock-position-tolerance 0.16

### Resume loaded delivery

Only after visually verifying cargo pickup:

    PYTHONPATH="$PWD/tools/warehouse_navigation:$PWD/tools/cargo_mission:$PYTHONPATH" python3 tools/cargo_mission/mission.py --pickup sw --dropoff nw --no-smoothing --resume-loaded --confirm-cargo

### Final drop-off and unloading

Only when the robot is carrying cargo at the northwest delivery approach:

    PYTHONPATH="$PWD/tools/warehouse_navigation:$PWD/tools/cargo_mission:$PYTHONPATH" python3 tools/cargo_mission/mission.py --pickup sw --dropoff nw --no-smoothing --resume-unload --confirm-cargo

These commands are state-dependent. Docking, retreat, and unloading must not be executed from arbitrary robot positions.

## Current Limitations

- Ground-truth localization is used in the demonstrated simulation.
- Cargo pickup and release require visual confirmation.
- Precision docking is calibrated for the southwest station.
- The mission currently runs in separate stages.
- Cargo-aware obstacle avoidance requires further validation.
- The experimental direction-lock node is not integrated.
- Physical robot deployment has not been validated.

## Future Development

- Unified autonomous mission execution
- Additional pickup and drop-off stations
- Automated cargo verification
- Improved docking accuracy
- Localization independent of Gazebo ground truth
- Cargo-aware obstacle avoidance
- Safety and recovery behavior

## License

Refer to the package-level license files where available.

<!-- PHASE14_MILESTONE_START -->
## Project Status — Phase 14: Two-Stage Precision Docking

The Autonomous Tugger V2 project is a ROS 2 and Gazebo simulation
of an autonomous four-wheel-steering warehouse tugger.

### Development Environment

- Ubuntu 26.04
- ROS 2 Lyrical
- Gazebo Sim 10
- Four-wheel-steering (4WS) mobile robot
- Warehouse navigation and simulated cargo stations

### Navigation and Docking

The current navigation system includes:

- Warehouse map and station configurations.
- Manhattan–Bézier route generation and saved route execution.
- Forward and reverse warehouse route candidates.
- Predocking route generation for four cargo stations.
- Two-stage docking with a mandatory stationary handoff.
- Low-speed precision entry with continuous heading and lateral correction.
- Odometry, TF, steering, and final-pose checks.

**Segment 1:** Navigate from HOME to a predocking location and stop.

**Segment 2:** Enter the cargo station while correcting position and
heading, then verify the final stationary pose.

The successfully tested southeast drop-off target is:

| Parameter | Target |
| --- | --- |
| X | 17.500 m |
| Y | -7.000 m |
| Heading | -90 degrees |
| Position tolerance | 0.02 m |
| Heading tolerance | 2 degrees |

### Phase 14 Gazebo Validation

The first supervised Gazebo precision-entry test completed successfully.

| Measurement | Observed result |
| --- | --- |
| Final X | Approximately 17.496 m |
| Final Y | Approximately -6.981 m |
| Final heading | Approximately -89.54 degrees |
| Final position error | Approximately 1.94 cm |
| Final heading error | Approximately 0.46 degrees |
| Final linear speed | 0.000 m/s |
| Final steering angle | Approximately 0 degrees |

The precision-entry controller reported successful stationary-pose
verification.

Offline tests also completed 125/125 dynamic controller cases and
125/125 modeled clearance cases with the selected correction gains.

These results demonstrate a working Gazebo navigation and docking
milestone, **not full collision-safety certification**.

### Current Controller Configuration

The Phase 14 docking controller uses:

- Heading correction gain: 4.0
- Lateral correction gain: 6.0
- Maximum commanded precision-entry speed: 0.08 m/s
- Final stationary-pose verification before reporting success

Temporary FAST_SIM settings are used for development. Normal-speed
operation will require separate validation.

### Safety and Current Limitations

- Cargo lifting remains disabled.
- Full articulated-wheel and robot 3D collision safety is not certified.
- The validated physical-motion run applies to the southeast drop-off.
- Other cargo stations still require supervised execution validation.
- Docking tolerances must be rechecked when speed settings change.

### Key Phase 14 Files

- `tools/v1_manhattan_bezier/phase14/phase14_nav_to_predock.py`
- `tools/v1_manhattan_bezier/phase14/phase14_precision_entry.py`
- `tools/v1_manhattan_bezier/phase14/check_predock.py`
- `tools/v1_manhattan_bezier/phase14/refine_docking_routes.py`
- `tools/v1_manhattan_bezier/phase14/routes_refined/`

### Development Status

**Phase 14 navigation and Gazebo precision docking: complete for the
tested southeast drop-off scenario.**

Next work will address cargo-handling safety and further autonomous
mission integration.
<!-- PHASE14_MILESTONE_END -->
