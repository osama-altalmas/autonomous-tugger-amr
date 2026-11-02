# AMR V2 — Restored Phase 14 Verification

**Date:** 2026-10-31
**Branch:** nav2-development
**Source baseline:** 7b13cbc

## Verified platform

- Gazebo simulation operational.
- Four-wheel-steering controller operational.
- Traction controller active.
- Steering controller active.
- Lift controller active.
- Joint-state broadcaster active.
- Ground-truth localization operational.
- Warehouse map server active.
- ROS velocity command connections verified.

## Phase 14D: Warehouse navigation

Route: HOME to cargo_dropoff_southeast

- Saved route: 981 poses.
- Footprint validation: PASSED.
- Route follower: GOAL REACHED.
- Final pre-docking pose:
  X = 17.502 m
  Y = -5.962 m
  Heading = -91.00 degrees
- Target checkpoint:
  X = 17.500 m
  Y = -6.000 m
  Heading = -90.00 degrees
- Checkpoint position error: approximately 3.8 cm.
- Checkpoint heading error: approximately 1 degree.
- Strict 2 cm position gate: NOT PASSED.
- Maximum commanded warehouse speed: 1.6 m/s.

Reduced-speed navigation validation is outstanding.

## Phase 14E: Live precision docking

Final docking target:

- X = 17.500 m
- Y = -7.000 m
- Heading = -90.00 degrees

Live Gazebo test:

- Read-only entry gate: PASSED.
- Precision-entry handoff: ACCEPTED.
- Maximum commanded docking speed: 0.08 m/s.
- Final reported remaining distance: approximately 1.8 cm.
- Final reported lateral error: approximately 1 mm.
- Final reported heading error: approximately 0.11 degrees.
- Final stationary-pose verification: PASSED.
- Controller exit code: 0.
- Cargo lift: DISABLED.

## Phase 14F: Offline testing

- Five-case 4WS dynamics test: 5/5 PASSED.
- Dynamic stress test: 118/125 PASSED.
- Seven stress cases remain outside acceptance criteria.

These offline results do not establish equivalent live
Gazebo performance.

## Outstanding work

1. Retest warehouse navigation at lower speed.
2. Confirm repeatable 2 cm checkpoint acceptance.
3. Resolve seven docking stress-test failures.
4. Reconcile offline and live docking gains.
5. Validate articulated-wheel 3D clearance.
6. Validate surrounding-obstacle clearance.
7. Validate guarded Segment 1-to-2 integration.
8. Validate cargo lifting and attachment separately.

## Safety scope

This checkpoint documents experimental Gazebo results.

- Full articulated-wheel clearance is not certified.
- Refined route execution_approved remains false.
- Automatic precision-entry authorization remains disabled
  in the Phase 14D launcher.
- Cargo handling remains unvalidated.
- Physical deployment safety is not certified.

## Conclusion

The restored Phase 14 code successfully demonstrated
HOME-to-southeast warehouse navigation and a separate,
successful precision-docking maneuver in Gazebo.

No experimental Phase 15 code is included in this
verification commit.
