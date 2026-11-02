#!/usr/bin/env python3
"""Phase 14E: guarded Gazebo-only slow precision entry, no lifting.

Default: read-only preflight. Motion requires BOTH --drive and --gazebo-only.
This is an experimental simulator controller, not certified collision avoidance.
"""
import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

JOINTS = ('front_left_steering_joint', 'front_right_steering_joint',
          'rear_left_steering_joint', 'rear_right_steering_joint')


def wrap(x):
    return math.atan2(math.sin(x), math.cos(x))


def corners(p, hx, hy):
    x, y, yaw = p
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x+c*dx-s*dy, y+s*dx+c*dy) for dx,dy in
            ((-hx,-hy),(hx,-hy),(hx,hy),(-hx,hy))]


def intersect(a, b):
    for poly in (a, b):
        for i, p in enumerate(poly):
            q = poly[(i+1)%len(poly)]
            axis = (p[1]-q[1],q[0]-p[0])
            da = [v[0]*axis[0]+v[1]*axis[1] for v in a]
            db = [v[0]*axis[0]+v[1]*axis[1] for v in b]
            if max(da) < min(db) or max(db) < min(da):
                return False
    return True


class Dock(Node):
    def __init__(self, data, drive):
        super().__init__('phase14_precision_entry')
        results = self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        if not all(r.successful for r in results):
            raise RuntimeError('Simulation time unavailable')
        self.drive = drive
        self.data = data
        self.target = data['final_docking_pose']
        self.predock = data['predocking_pose']
        self.legs = data['geometry']['stand_legs']
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.odom = None
        self.joints = None
        self.odom_t = None
        self.joints_t = None
        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10) if drive else None
        self.create_subscription(Odometry, '/odometry/filtered', self.got_odom, 10)
        self.create_subscription(JointState, '/joint_states', self.got_joints, 10)
        self.leg_polys = [corners((a['x'], a['y'], a['yaw']),
                                  a['size'][0]/2, a['size'][1]/2) for a in self.legs]
        self.started = False
        self.start_wall = None
        self.aligned_count = 0
        self.passed = False
        self.reason = 'NOT STARTED'
        self.last_log = 0

    def got_odom(self, msg):
        self.odom = msg
        self.odom_t = time.monotonic()

    def got_joints(self, msg):
        self.joints = dict(zip(msg.name, msg.position))
        self.joints_t = time.monotonic()

    def publish(self, v=0., omega=0.):
        if self.pub is None or not rclpy.ok():
            return
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(omega)
        self.pub.publish(msg)

    def current(self):
        now = time.monotonic()
        if self.odom is None or now-self.odom_t > 0.7:
            raise ValueError('Missing/stale odometry')
        if self.joints is None or now-self.joints_t > 0.7:
            raise ValueError('Missing/stale joint states')
        missing = [n for n in JOINTS if n not in self.joints]
        if missing:
            raise ValueError('Missing measured steering joints: '+str(missing))
        tr = self.tf.lookup_transform('map', 'base_footprint', Time())
        stamp = tr.header.stamp
        age = (self.get_clock().now().nanoseconds -
               (stamp.sec*10**9+stamp.nanosec))/1e9
        if age < -.1 or age > .35:
            raise ValueError(f'Stale map TF ({age:.3f} s)')
        q = tr.transform.rotation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y),
                         1-2*(q.y*q.y+q.z*q.z))
        t = tr.transform.translation
        return (t.x,t.y,yaw)

    def geometry_ok(self, pose):
        # A conservative envelope covering the 0.88 x 0.58 m chassis
        # and nominal steering wheels, plus margin. It is NOT a full
        # articulated 3D wheel-sweep certificate.
        body = corners(pose, .49, .355)
        return not any(intersect(body, leg) for leg in self.leg_polys)

    def assessed(self, pose):
        x, y, yaw = pose
        gx,gy,gh = self.target
        ux,uy = math.cos(gh), math.sin(gh)
        dx,dy = x-gx,y-gy
        remaining = -(dx*ux+dy*uy)
        lateral = -uy*dx+ux*dy
        heading_err = wrap(gh-yaw)
        distance = math.hypot(dx,dy)
        return remaining,lateral,heading_err,distance

    def gate(self, pose):
        rem, lat, hd, _ = self.assessed(pose)
        if not (-.01 <= rem <= 1.25):
            raise ValueError(f'Outside entry longitudinal gate: {rem:.3f} m')
        if abs(lat) > .09:
            raise ValueError(f'Outside entry corridor: lateral={lat:.3f} m')
        if abs(hd) > math.radians(8):
            raise ValueError(f'Heading exceeds entry gate: {math.degrees(hd):.2f} deg')
        if not self.geometry_ok(pose):
            raise ValueError('Conservative robot envelope intersects station leg')
        # Check prospective approach aligned to actual measured heading.
        for d in (.03,.06,.10,.15):
            ahead = (pose[0]+d*math.cos(pose[2]),
                     pose[1]+d*math.sin(pose[2]),pose[2])
            if not self.geometry_ok(ahead):
                raise ValueError('Forward projected envelope intersects a leg')

    def tick(self):
        try:
            pose = self.current()
            self.gate(pose)
            rem,lat,hd,dist = self.assessed(pose)
            v_twist = self.odom.twist.twist
            speed = math.hypot(v_twist.linear.x,v_twist.linear.y)
            steering = max(abs(self.joints[j]) for j in JOINTS)
            if not self.started:
                # Must be truly stationary before engaging second segment.
                if speed > .025 or abs(v_twist.angular.z) > .025 or steering > math.radians(3):
                    raise ValueError('Robot not stationary/steering not straight at handoff')
                self.started = True
                self.start_wall = time.monotonic()
                print(f'HANDOFF ACCEPTED: pose={pose}; lateral={lat:.3f} m; heading={math.degrees(hd):.2f} deg',flush=True)
            if time.monotonic()-self.start_wall > 90:
                raise ValueError('Precision entry timed out')
            goal = (dist <= .02 and abs(hd) <= math.radians(2))
            if goal:
                # Stop before accepting the final docking pose.
                self.publish()

                stationary = (
                    speed <= .01
                    and abs(v_twist.angular.z) <= .02
                    and steering <= math.radians(2)
                )

                if stationary:
                    if getattr(self, 'stable_since', None) is None:
                        self.stable_since = time.monotonic()
                    self.aligned_count += 1
                else:
                    self.stable_since = None
                    self.aligned_count = 0

                stable_duration = (
                    time.monotonic() - self.stable_since
                    if self.stable_since is not None
                    else 0.0
                )

                if self.aligned_count >= 10 and stable_duration >= 1.0:
                    self.passed = True
                    self.reason = (
                        'FINAL STATIONARY POSE VERIFIED — '
                        'CARGO OPERATION STILL DISABLED'
                    )
                    return True
            else:
                # Pose must stay within tolerance after braking.
                self.stable_since = None
                self.aligned_count = 0
                if rem < .015:
                    raise ValueError('Reached center longitudinally without passing pose gate')
                # 4WS counter-phase follows forward arc; cannot crab or pivot.
                # Lateral correction and heading alignment both converge
                # toward the centerline, with steering geometry constraints.
                # Body heading error is target_heading - current_heading.
                curvature = 4.0*hd - 6.0*lat
                curvature = max(-.9,min(.9,curvature))
                v = min(.08, max(.018, .20*rem))
                if rem < .07:
                    v = min(v,.025)
                omega = v*curvature
                self.publish(v,omega)
            now = time.monotonic()
            if now-self.last_log > 1:
                self.last_log = now
                print(f'entry remaining={rem:.3f}m lateral={lat:.3f}m heading_error={math.degrees(hd):.2f}deg speed={speed:.3f}m/s steering={math.degrees(steering):.2f}deg',flush=True)
            return False
        except Exception as exc:
            self.reason = 'ENTRY ABORTED: '+str(exc)
            self.publish()
            return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--route',type=Path,required=True)
    ap.add_argument('--drive',action='store_true')
    ap.add_argument('--gazebo-only',action='store_true')
    args = ap.parse_args()
    if args.drive != args.gazebo_only:
        ap.error('Motion requires BOTH --drive and --gazebo-only')
    data = json.loads(args.route.read_text())
    if data.get('schema') != 'amr_phase14_two_segment_refined_v1':
        raise SystemExit('Refined Phase 14B route required')
    if data.get('execution_approved') is not False:
        raise SystemExit('Unexpected route execution flag')
    if data.get('validation',{}).get('2d_leg_geometry') != 'PASSED':
        raise SystemExit('Leg geometry validation not passed')
    if data.get('validation',{}).get('static_vertical_body_lift_clearance') not in ('PASSED','NOT_APPLICABLE_NO_TABLETOP_COLLISION'):
        raise SystemExit('Static height geometry validation not passed')
    print('PHASE 14E: EXPERIMENTAL PRECISION ENTRY — NO CARGO LIFT')
    print('Target:',data['final_docking_pose'])
    print('Dynamic articulated-wheel 3D sweep: NOT CERTIFIED')
    print('Mode:', 'GAZEBO MOTION' if args.drive else 'READ-ONLY ASSESSMENT',flush=True)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Dock(data,args.drive)
    complete = False
    start = time.monotonic()

    # ROS subscriptions and TF need time to initialize.
    # Never command movement while feedback is unavailable.
    feedback_ready = False
    last_wait_log = 0.0

    try:
        while rclpy.ok() and time.monotonic()-start < (100 if args.drive else 12):
            rclpy.spin_once(node, timeout_sec=.05)

            if args.drive:
                if not feedback_ready:
                    try:
                        node.current()
                        feedback_ready = True
                        print(
                            "INITIAL FEEDBACK READY — "
                            "beginning guarded entry assessment",
                            flush=True
                        )
                    except Exception as exc:
                        elapsed = time.monotonic() - start

                        if elapsed - last_wait_log >= 1.0:
                            last_wait_log = elapsed
                            print(
                                f"Waiting for feedback ({elapsed:.1f}s): "
                                f"{exc}",
                                flush=True
                            )

                        if elapsed > 10.0:
                            node.reason = (
                                "ENTRY ABORTED: Startup feedback "
                                "unavailable after 10 seconds"
                            )
                            break

                        continue

                complete = node.tick()
                if complete:
                    break
            else:
                try:
                    pose=node.current()
                    node.gate(pose)
                    print('ENTRY GATE PASSED (read-only):',pose,'metrics=',node.assessed(pose),flush=True)
                    break
                except Exception as exc:
                    if time.monotonic()-start > 4:
                        print('ENTRY GATE NOT READY:',exc,flush=True)
                        break
    except KeyboardInterrupt:
        node.reason = 'Operator interrupted'
    finally:
        if args.drive:
            for _ in range(12):
                node.publish()
                rclpy.spin_once(node,timeout_sec=.05)
        print('RESULT:',node.reason if args.drive else 'READ-ONLY COMPLETE',flush=True)
        print('LIFT: LOCKED',flush=True)
        node.destroy_node()
        rclpy.shutdown()
    return 0 if (node.passed if args.drive else True) else 2

if __name__ == '__main__':
    sys.exit(main())
