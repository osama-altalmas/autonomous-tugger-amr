#!/usr/bin/env python3
"""Offline, static-map graph mission for a tugger starting at the warehouse origin.

Uses the existing SDF collision validator and graph generator; never queries
Gazebo ground-truth obstacle positions at runtime. This is an initial-pose
mission, NOT a general arbitrary-pose graph connector.
"""
import argparse
import heapq
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
WORKSPACE = HERE.parents[2]
sys.path.insert(0, str(WORKSPACE / 'tools/warehouse_validation'))
import generate_route_network as planner


def candidates(adjacency, start, goal):
    queue = [(0.0, (start,))]
    while queue:
        cost, seq = heapq.heappop(queue)
        if seq[-1] == goal:
            yield cost, seq
            continue
        for neighbor, weight in adjacency[seq[-1]]:
            if neighbor not in seq:
                heapq.heappush(queue, (cost + weight, seq + (neighbor,)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--goal', default='H2_1', help='Existing graph node')
    p.add_argument('--entry', default='H1_2', help='Graph node reached from (0,0)')
    p.add_argument('--smoothing', type=float, default=0.8)
    p.add_argument('--max-steering-deg', type=float, default=42.0,
                   help='Extra headroom below physical 45-degree limit')
    p.add_argument('--output', default=str(WORKSPACE / 'tools/warehouse_validation/results/route_graph_mission.json'))
    args = p.parse_args()
    if args.smoothing <= 0 or not (0 < args.max_steering_deg <= 45):
        p.error('Invalid smoothing or steering limit')

    occupied, protected = planner.build_occupancy()
    nodes, adjacency, edges = planner.build_graph(protected)
    if args.entry not in nodes or args.goal not in nodes or args.entry == args.goal:
        p.error('Entry and goal must be different existing graph nodes')

    origin = [0.0, 0.0]
    entry = nodes[args.entry]
    # Explicitly limit this batch to the validated straight connector along +X.
    if abs(entry[1]) > 1e-8 or entry[0] <= 0:
        p.error('This initial-pose mission requires an entry on y=0, x>0')
    connector = planner.sample_line(origin, entry)
    if not planner.collision_free(connector, protected):
        raise SystemExit('STOP: initial connector is not collision-free')

    chosen = None
    evaluated = 0
    for cost, sequence in candidates(adjacency, args.entry, args.goal):
        evaluated += 1
        if evaluated > 2000:
            break
        waypoints = [origin] + [nodes[name] for name in sequence]
        path, error = planner.smooth_route(waypoints, args.smoothing)
        if path is None or not planner.collision_free(path, protected):
            continue
        feasible, curvature, steering = planner.check_kinematics(path)
        if not feasible or math.degrees(steering) > args.max_steering_deg:
            continue
        poses = planner.make_poses(path, 'forward')
        if abs(poses[0][0]) > 1e-6 or abs(poses[0][1]) > 1e-6 or abs(poses[0][2]) > 1e-5:
            continue
        chosen = {
            'nodes': ['INITIAL_POSE'] + list(sequence),
            'direction': 'forward',
            'poses': poses,
            'graph_length_m': round(cost + math.dist(origin, entry), 3),
            'path_length_m': round(sum(math.dist(a, b) for a, b in zip(path, path[1:])), 3),
            'max_curvature': round(curvature, 5),
            'max_steering_deg': round(math.degrees(steering), 3),
            'routes_evaluated': evaluated,
        }
        break
    if chosen is None:
        raise SystemExit('STOP: no collision-free, steering-feasible route found')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        'world': 'tugger_test',
        'source': 'warehouse SDF static collision geometry',
        'robot_clearance_radius_m': round(planner.CLEARANCE, 4),
        'smoothing_distance_m': args.smoothing,
        'selected_start': 'INITIAL_POSE',
        'selected_goal': args.goal,
        'driving_direction': 'forward',
        'nodes': nodes,
        'edges': edges,
        'selected_route': chosen,
        'limitations': [
            'Initial pose fixed at map (0,0,0); executor checks actual TF alignment',
            'Static SDF clearance uses conservative circular robot envelope',
            'No measured steering-rate or swept-trailer validation',
            'Dynamic obstacles require existing Nav2 collision protection',
        ],
    }
    out.write_text(json.dumps(result, indent=2))
    print('GRAPH MISSION: VALIDATED (offline static geometry + curvature)')
    print('Route:', ' -> '.join(chosen['nodes']))
    print('Length:', chosen['path_length_m'], 'm')
    print('Max steering:', chosen['max_steering_deg'], 'deg')
    print('Candidate routes evaluated:', evaluated)
    print('Saved:', out)
    print('NEXT: inspect in RViz and confirm actual warehouse clearance before --execute')


if __name__ == '__main__':
    main()
