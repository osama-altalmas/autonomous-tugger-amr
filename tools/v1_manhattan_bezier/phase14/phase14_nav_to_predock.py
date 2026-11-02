#!/usr/bin/env python3
"""Phase 14D: execute ONLY the refined warehouse segment to predock.

Uses the existing V1 follower unchanged. No precision entry or lift commands.
Default is offline validation + ROS preflight, NOT motion.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
TOOL_DIR = ROOT / 'tools/v1_manhattan_bezier'
sys.path.insert(0, str(TOOL_DIR))
from follow_saved_route_v1 import (  # noqa: E402
    SavedRouteFollower, current_map_hash, load_and_validate, wrap, distance
)
from build_network import load_occupancy, validate_trajectory  # noqa: E402
import rclpy  # noqa: E402
from rclpy.signals import SignalHandlerOptions

DB = TOOL_DIR / 'routes/route_database.json'
MAP = ROOT / 'src/tugger_navigation/maps/warehouse_map.yaml'
PROFILE = ROOT / 'src/tugger_control/config/sim_speed.yaml'
REFINED = TOOL_DIR / 'phase14/routes_refined'
MONITOR = TOOL_DIR / 'phase14/check_predock.py'


def verify(route_path, source, target):
    refined = json.loads(route_path.read_text())
    if refined.get('schema') != 'amr_phase14_two_segment_refined_v1':
        raise ValueError('Not a Phase 14B refined route')
    if refined.get('execution_approved') is not False:
        raise ValueError('Refined route must explicitly deny docking authorization')
    if refined.get('source') != source or refined.get('destination') != target:
        raise ValueError('Refined route source/destination mismatch')
    if refined.get('map_sha256') != current_map_hash(MAP):
        raise ValueError('Refined route map fingerprint mismatch')
    checks = refined.get('validation', {})
    if checks.get('2d_leg_geometry') != 'PASSED':
        raise ValueError('Leg-collision validation has not passed')
    if checks.get('static_vertical_body_lift_clearance') not in (
        'PASSED', 'NOT_APPLICABLE_NO_TABLETOP_COLLISION'
    ):
        raise ValueError('Static body/lift height validation has not passed')

    checkpoint = refined['checkpoint']['target_pose']
    first = refined['segment_1']['poses']
    second = refined['segment_2']['poses']
    if len(first) < 3 or len(second) < 2:
        raise ValueError('Segment poses missing')
    if distance(first[-1], checkpoint) > 1e-6:
        raise ValueError('Segment 1 does not end at checkpoint')
    if distance(second[0], checkpoint) > 1e-6:
        raise ValueError('Segment 2 is not continuous at checkpoint')
    if abs(wrap(first[-1][2] - checkpoint[2])) > 1e-6:
        raise ValueError('Segment 1 endpoint heading mismatch')
    if any(len(p) != 3 or not all(math.isfinite(v) for v in p)
           for p in first):
        raise ValueError('Nonfinite or malformed segment 1 pose')

    # Leverage the original follower's full database, hash, heading,
    # occupancy, and static-footprint validation of the saved route.
    fake = argparse.Namespace(database=str(DB), map=str(MAP),
                              source=source, target=target)
    original, direction, original_poses = load_and_validate(fake)
    if direction != 'FORWARD':
        raise ValueError('Phase 14D currently supports forward approach only')
    if len(first) <= len(original_poses):
        raise ValueError('Refined Segment 1 is missing its staging extension')
    for i, (a, b) in enumerate(zip(first, original_poses)):
        if distance(a, b) > 1e-5 or abs(wrap(a[2] - b[2])) > 1e-5:
            raise ValueError(f'Original saved warehouse route modified at {i}')

    # Crucially recheck the ENTIRE extended Segment 1 against the map.
    # The station-entry segment is *not* checked against 2D obstacles
    # because it would enter under the tabletop, and is never driven here.
    spec, blocked = load_occupancy(MAP)
    result = validate_trajectory(first, spec, blocked)
    if not result['valid']:
        raise ValueError('Refined Segment 1 fails full occupancy-map footprint check')
    worst = 0.0
    for a, b in zip(first, first[1:]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        if math.hypot(dx, dy) > 1e-7:
            worst = max(worst, abs(wrap(math.atan2(dy, dx) - a[2])))
    if worst > math.radians(12):
        raise ValueError(f'Segment 1 heading/trajectory mismatch: {math.degrees(worst):.2f} degrees')
    return refined, first, direction


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--to', default='cargo_dropoff_southeast')
    ap.add_argument('--route', type=Path, default=None)
    ap.add_argument('--check-only', action='store_true',
                    help='Offline static validation, no ROS nodes')
    ap.add_argument('--drive', action='store_true',
                    help='Explicitly permit Segment 1 movement in Gazebo')
    ap.add_argument('--gazebo-only', action='store_true',
                    help='Second mandatory motion enable gate')
    args = ap.parse_args()
    if args.drive != args.gazebo_only:
        ap.error('Movement requires BOTH --drive and --gazebo-only')
    path = args.route or REFINED / f'HOME__TO__{args.to}__REFINED.json'
    refined, poses, direction = verify(path, 'HOME', args.to)
    print('\n=== PHASE 14D: SEGMENT 1 ONLY ===', flush=True)
    print(f'Station: {args.to}\nSegment 1 poses: {len(poses)}')
    print('Checkpoint:', refined['checkpoint']['target_pose'])
    print('Original + extended map footprint validation: PASSED')
    print('Segment 2: LOCKED; lift: LOCKED; motion:',
          'ENABLED (GAZEBO)' if args.drive else 'DISABLED', flush=True)
    if args.check_only:
        print('OFFLINE CHECK PASSED — robot not moved')
        return 0

    with PROFILE.open() as f:
        config = yaml.safe_load(f)
    ctl = config['four_ws_controller']['ros__parameters']
    follower = config['follower']
    speed = float(follower['cruising_speed'])
    lateral = float(follower['max_lateral_acceleration'])
    if not (0 < speed <= float(ctl['max_linear_speed']) <= 2.5):
        raise ValueError('Follower/controller speed profile mismatch')
    if not (0 < lateral <= float(ctl['max_lateral_acceleration'])):
        raise ValueError('Lateral acceleration profile mismatch')

    fargs = argparse.Namespace(
        enable_motion=args.drive, gazebo_only=args.gazebo_only,
        resume=False, speed=speed, lookahead=float(follower['lookahead']),
        max_tracking_error=float(follower['max_tracking_error']),
        start_tolerance=0.20, start_yaw_deg=8.0,
        goal_tolerance=0.04, goal_yaw_deg=3.0,
        braking_margin=0.0,
        tf_timeout=0.40, max_seconds=float(follower['max_seconds']),
        wheelbase=0.58, max_steering=math.pi/4,
        max_lateral_accel=lateral,
        braking_deceleration=float(ctl['max_deceleration'])
    )
    # Let Python handle Ctrl+C so ROS remains initialized
    # while we publish repeated zero-velocity commands.
    rclpy.init(
        signal_handler_options=SignalHandlerOptions.NO
    )
    node = SavedRouteFollower(fargs, poses, direction)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.10)
    except KeyboardInterrupt:
        node.stop('Operator requested stop', failed=True)
    finally:
        if node.motion_enabled:
            for _ in range(10):
                node.command(0.0, 0.0)
                rclpy.spin_once(node, timeout_sec=0.05)
        failed = node.failed_reason is not None
        reached = node.finished and not failed and node.phase == 'FOLLOWING'
        node.destroy_node()
        rclpy.shutdown()

    if failed:
        print('SEGMENT 1 FAILED. Docking remains locked.', flush=True)
        return 2
    if not args.drive:
        print('PREFLIGHT COMPLETE. No robot motion.', flush=True)
        return 0
    if not reached:
        print('Segment 1 completion not confirmed. Docking locked.')
        return 2

    print('\nSEGMENT 1 FOLLOWER FINISHED. Checking stop/alignment...', flush=True)
    print('No automatic correction or precision entry will be attempted.', flush=True)
    # Read-only monitor; result is informational, NOT docking permission.
    rc = subprocess.call([
        sys.executable, '-u', str(MONITOR), '--route', str(path),
        '--duration', '15',
    ], cwd=ROOT)
    print('PHASE 14D ENDED — Segment 2 and lift remain LOCKED.', flush=True)
    return rc


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError) as exc:
        print('PHASE 14D REFUSED:', exc, file=sys.stderr)
        sys.exit(2)
