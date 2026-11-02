#!/usr/bin/env python3
"""Live-pose graph routing with validated straight entry and optional FollowPath execution.
No privileged Gazebo obstacle access; uses static warehouse SDF for offline clearance.
"""
import argparse
import heapq
import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve()
WS=HERE.parents[2]
sys.path.insert(0,str(WS/'tools/warehouse_validation'))
import generate_route_network as planner
from warehouse_alignment import WarehouseAlignment
from check_warehouse_alignment import gazebo_pose


def angle(a):
    return math.atan2(math.sin(a),math.cos(a))

def routes(adj,start,goal,limit=2000):
    queue=[(0.,(start,))]
    attempts=0
    while queue and attempts<limit:
        cost,seq=heapq.heappop(queue)
        if seq[-1]==goal:
            attempts+=1
            yield cost,seq
            continue
        for nxt,w in adj[seq[-1]]:
            if nxt not in seq:
                heapq.heappush(queue,(cost+w,seq+(nxt,)))

def plan(x,y,yaw,goal,smoothing=0.8,steering_limit=42,entry_tolerance_deg=12):
    occupied,protected=planner.build_occupancy()
    nodes,adj,edges=planner.build_graph(protected)
    if goal not in nodes:
        raise ValueError(f"Unknown destination {goal}; choose: {', '.join(sorted(nodes))}")
    start=[float(x),float(y)]
    if not planner.collision_free([start],protected):
        raise ValueError("Current pose is inside static protected obstacle zone")
    candidates=[]
    # A direct, heading-compatible connection to an existing graph node.
    # Do not teleport, rotate in place, or assume a safe sideways connector.
    for entry,xy in nodes.items():
        d=math.dist(start,xy)
        if d<0.10:
            # Already at a graph node; first actual graph edge sets heading.
            candidates.append((0.0,entry,'forward'))
            candidates.append((0.0,entry,'reverse'))
            continue
        bearing=math.atan2(xy[1]-y,xy[0]-x)
        for direction in ('forward','reverse'):
            heading=angle(bearing+(math.pi if direction=='reverse' else 0))
            err=abs(angle(yaw-heading))
            if err>math.radians(entry_tolerance_deg):
                continue
            if planner.collision_free(planner.sample_line(start,xy),protected):
                candidates.append((d,entry,direction))
    if not candidates:
        raise ValueError("No straight, heading-compatible, collision-free graph entry. "
                         "Stop; a nonholonomic curved connector is needed here.")
    solutions=[]
    for connector_len,entry,direction in sorted(candidates):
        for graph_len,seq in routes(adj,entry,goal):
            waypoints=([start] if connector_len>=0.10 else [])+[nodes[n] for n in seq]
            if len(waypoints)<2:
                continue
            path,error=planner.smooth_route(waypoints,smoothing)
            if path is None or not planner.collision_free(path,protected):
                continue
            valid,kappa,steer=planner.check_kinematics(path)
            if not valid or math.degrees(steer)>steering_limit:
                continue
            poses=planner.make_poses(path,direction)
            if abs(angle(poses[0][2]-yaw))>math.radians(entry_tolerance_deg):
                continue
            length=sum(math.dist(a,b) for a,b in zip(path,path[1:]))
            solutions.append((length,{
                'nodes':['LIVE_POSE']+list(seq),
                'direction':direction,
                'poses':poses,
                'graph_length_m':round(connector_len+graph_len,3),
                'path_length_m':round(length,3),
                'max_curvature':round(kappa,5),
                'max_steering_deg':round(math.degrees(steer),3),
                'entry_node':entry,
                'entry_distance_m':round(connector_len,3),
            }))
            # For this entry/direction the first valid candidate is sufficient.
            break
    if not solutions:
        raise ValueError("No feasible route: entry, curvature, or clearance rejected")
    length,chosen=min(solutions,key=lambda t:t[0])
    return {
        'world':'tugger_test',
        'source':'static warehouse SDF + live map->base_link TF',
        'selected_start':'LIVE_POSE',
        'selected_goal':goal,
        'driving_direction':chosen['direction'],
        'robot_clearance_radius_m':round(planner.CLEARANCE,4),
        'smoothing_distance_m':smoothing,
        'selected_route':chosen,
        'limitations':[
            'Only straight, heading-compatible entry connectors; no arbitrary-pose nonholonomic connector',
            'Static SDF clearance; live dynamic obstacle handling remains Nav2 responsibility',
            'No steering-rate or trailer swept-path validation',
            'A single forward or reverse direction is used throughout each mission',
            'Route is shortest among accepted candidates, not globally optimal over all maneuvers'
        ]
    }

def live_pose(frame,base,timeout):
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from tf2_ros import Buffer,TransformListener,TransformException
    rclpy.init()
    node=Node('warehouse_live_graph_pose',parameter_overrides=[
        Parameter('use_sim_time',Parameter.Type.BOOL,True)])
    buffer=Buffer()
    listener=TransformListener(buffer,node)
    deadline=time.monotonic()+timeout
    count=0
    previous_stamp=None
    try:
        while time.monotonic()<deadline and rclpy.ok():
            rclpy.spin_once(node,timeout_sec=0.2)
            try:
                tf=buffer.lookup_transform(frame,base,rclpy.time.Time())
                stamp=(tf.header.stamp.sec,tf.header.stamp.nanosec)
                if stamp==previous_stamp:
                    continue
                previous_stamp=stamp
                count+=1
                if count>=5:
                    t=tf.transform.translation
                    q=tf.transform.rotation
                    yaw=math.atan2(2*(q.w*q.z+q.x*q.y),
                                   1-2*(q.y*q.y+q.z*q.z))
                    return t.x,t.y,yaw
            except TransformException:
                count=0
        raise RuntimeError('Timed out waiting for five fresh TF samples')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--goal',required=True,help='Warehouse graph node, e.g. H2_1')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--smoothing',type=float,default=0.8)
    parser.add_argument('--max-steering-deg',type=float,default=42.)
    parser.add_argument('--entry-heading-tolerance',type=float,default=12.)
    parser.add_argument('--frame',default='map')
    parser.add_argument('--base-frame',default='base_link')
    parser.add_argument('--tf-timeout',type=float,default=90.)
    parser.add_argument('--output',default=str(WS/'tools/warehouse_validation/results/route_live_mission.json'))
    parser.add_argument('--test-pose',nargs=3,type=float,metavar=('X','Y','YAW_DEG'),
                        help='OFFLINE TEST ONLY; no execution permitted')
    args=parser.parse_args()
    if args.smoothing<=0 or not (0<args.max_steering_deg<=45):
        parser.error('Invalid smoothing or steering limit')
    if args.test_pose and args.execute:
        parser.error('Cannot execute with --test-pose')
    if args.test_pose:
        x,y,degrees=args.test_pose
        yaw=math.radians(degrees)
        print('OFFLINE TEST POSE; not connected to robot')
    else:
        print('Waiting for live TF map -> base_link...')
        map_pose = live_pose(args.frame,args.base_frame,args.tf_timeout)
        world_pose = gazebo_pose()
        alignment = WarehouseAlignment(world_pose, map_pose)
        x,y,yaw = alignment.map_to_world(*map_pose)
        print(
            f'Alignment: rotation={math.degrees(alignment.rotation):.3f} deg, '
            f'translation=({alignment.tx:.3f}, {alignment.ty:.3f})'
        )
    print(f'Robot start: x={x:.3f}, y={y:.3f}, yaw={math.degrees(yaw):.1f} deg')
    result=plan(x,y,yaw,args.goal,args.smoothing,args.max_steering_deg,
                args.entry_heading_tolerance)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if not args.test_pose:
        result['coordinate_frame'] = 'warehouse_world'
        result['world_to_map_alignment'] = {
            'rotation_rad': alignment.rotation,
            'tx': alignment.tx,
            'ty': alignment.ty,
        }
    output.write_text(json.dumps(result,indent=2))
    route=result['selected_route']
    print('ROUTE:', ' -> '.join(route['nodes']))
    print('Direction:',route['direction'])
    print('Length:',route['path_length_m'],'m')
    print('Max steering:',route['max_steering_deg'],'deg')
    print('Saved:',output)
    if args.test_pose:
        return
    print('Starting existing FollowPath executor ('+
          ('EXECUTE' if args.execute else 'PREVIEW')+')')
    command=[sys.executable,str(WS/'tools/warehouse_navigation/follow_warehouse_route.py'),
             '--route',str(output),'--frame',args.frame,'--base-frame',args.base_frame]
    if args.execute:
        print('WARNING: only run after reviewing the planned route and clearance in RViz.')
        command.append('--execute')
    raise SystemExit(subprocess.call(command))

if __name__=='__main__':
    try:
        main()
    except (ValueError,RuntimeError) as e:
        print('STOP:',e,file=sys.stderr)
        raise SystemExit(2)
