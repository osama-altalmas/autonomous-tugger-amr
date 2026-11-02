#!/usr/bin/env python3
"""Autonomous tugger mission. Experimental docking requires explicit calibration."""
import argparse
import math
import sys
import time
from pathlib import Path

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path as NavPath
from nav2_msgs.action import ComputePathToPose, SmoothPath, FollowPath
from std_msgs.msg import Float64MultiArray
from tf2_ros import Buffer, TransformListener

# Existing warehouse/world coordinate alignment code is loaded by launcher via PYTHONPATH.
from warehouse_alignment import WarehouseAlignment
from path_direction_validator import validate_path_direction
from check_warehouse_alignment import gazebo_pose

STATIONS = {
    'sw': (-17.5, -7.0),
    'ne': (17.5, 7.0),
}
DROPS = {
    'nw': (-17.5, 7.0),
    'se': (17.5, -7.0),
}


def yaw_of(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))


def angle_diff(a,b):
    return math.atan2(math.sin(a-b), math.cos(a-b))


class Mission(Node):
    def __init__(self, args):
        super().__init__('tugger_cargo_mission', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.args=args
        self.tf=Buffer()
        self.listener=TransformListener(self.tf,self)
        self.plan=ActionClient(self,ComputePathToPose,'/compute_path_to_pose')
        self.smooth=ActionClient(self,SmoothPath,'/smooth_path')
        self.follow=ActionClient(self,FollowPath,'/follow_path')
        self.path_pub=self.create_publisher(NavPath,'/mission/planned_path',10)
        self.vel_pub=self.create_publisher(TwistStamped,'/cmd_vel',10)
        self.lift_pub=self.create_publisher(Float64MultiArray,'/lift_controller/commands',10)
        self.odom=None
        self.odom_time=0.0
        self.create_subscription(Odometry,'/wheel/odom',self.on_odom,10)
        self.alignment=None

    def on_odom(self,msg):
        self.odom=msg
        self.odom_time=time.monotonic()

    def spin_until(self,fn,seconds=30):
        deadline=time.monotonic()+seconds
        while rclpy.ok() and time.monotonic()<deadline:
            rclpy.spin_once(self,timeout_sec=0.1)
            if fn(): return True
        return False

    def robot_pose(self):
        t=self.tf.lookup_transform('map','base_footprint',rclpy.time.Time())
        p=t.transform.translation
        return (p.x,p.y,yaw_of(t.transform.rotation))

    def init_alignment(self):
        def ready():
            try:
                self.robot_pose()
                return True
            except Exception:
                return False
        if not self.spin_until(ready,45):
            raise RuntimeError('No map -> base_footprint TF. Check SLAM/odometry.')
        # Gazebo pose is used rather than assuming map and world stay aligned.
        world=gazebo_pose()
        map_pose=self.robot_pose()
        self.alignment=WarehouseAlignment(world,map_pose)
        self.get_logger().info(f'World/map alignment: translation ({self.alignment.tx:.3f}, {self.alignment.ty:.3f})')

    def pose(self,x,y,yaw):
        msg=PoseStamped()
        msg.header.frame_id='map'
        msg.header.stamp=self.get_clock().now().to_msg()
        msg.pose.position.x=x
        msg.pose.position.y=y
        msg.pose.orientation.z=math.sin(yaw/2)
        msg.pose.orientation.w=math.cos(yaw/2)
        return msg

    def send(self,client,goal,timeout=180):
        if not client.wait_for_server(timeout_sec=10):
            raise RuntimeError('Required Nav2 action server unavailable')
        future=client.send_goal_async(goal)
        if not self.spin_until(lambda:future.done(),20):
            raise RuntimeError('Nav2 goal acceptance timeout')
        handle=future.result()
        if not handle.accepted:
            raise RuntimeError('Nav2 action goal rejected')
        result=handle.get_result_async()
        if not self.spin_until(lambda:result.done(),timeout):
            handle.cancel_goal_async()
            raise RuntimeError('Nav2 action timed out and was cancelled')
        output=result.result()
        if output.status!=GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(f'Nav2 action failed with status {output.status}')
        return output.result

    def navigate(self,wx,wy,wyaw,stage):
        x,y,yaw=self.alignment.world_to_map(wx,wy,wyaw)
        goal=self.pose(x,y,yaw)
        self.get_logger().info(f'{stage}: planning map target ({x:.2f},{y:.2f},{yaw:.2f})')
        req=ComputePathToPose.Goal()
        req.goal=goal
        req.planner_id='GridBased'
        req.use_start=False
        result=self.send(self.plan,req,timeout=90)
        path=result.path
        if len(path.poses)<2:
            raise RuntimeError('Planner produced empty/degenerate path')

        try:
            planned_direction = validate_path_direction(path)
        except ValueError as exc:
            raise RuntimeError(f'{stage}: invalid planned direction: {exc}')

        self.get_logger().info(
            f'{stage}: original path is {planned_direction.upper()}-ONLY'
        )

        if self.args.smoothing:
            smooth=SmoothPath.Goal()
            smooth.path=path
            smooth.smoother_id='simple_smoother'
            smooth.max_smoothing_duration=Duration(seconds=5).to_msg()
            smooth.check_for_collisions=True
            try:
                candidate=self.send(self.smooth,smooth,timeout=25).path
                if len(candidate.poses)>=2:
                    path=candidate
                    self.get_logger().info('Nav2 collision-checked path smoothing succeeded')
            except RuntimeError as exc:
                if self.args.require_smoothing:
                    raise
                self.get_logger().warning(f'Smoothing unavailable; using original Nav2 path: {exc}')
        try:
            final_direction = validate_path_direction(path)
        except ValueError as exc:
            raise RuntimeError(f'{stage}: invalid final path direction: {exc}')

        if final_direction != planned_direction:
            raise RuntimeError(
                f'{stage}: smoothing changed driving direction '
                f'from {planned_direction} to {final_direction}'
            )

        self.get_logger().info(
            f'{stage}: validated {final_direction.upper()}-ONLY path'
        )

        self.path_pub.publish(path)
        self.get_logger().info(f'{stage}: path with {len(path.poses)} poses generated')
        if self.args.preview:
            return
        follow=FollowPath.Goal()
        follow.path=path
        follow.controller_id='FollowPath'
        follow.goal_checker_id='goal_checker'
        self.send(self.follow,follow,timeout=self.args.nav_timeout)
        self.get_logger().info(f'{stage}: navigation succeeded')

    def command(self,speed):
        msg=TwistStamped()
        msg.header.stamp=self.get_clock().now().to_msg()
        msg.header.frame_id='base_link'
        msg.twist.linear.x=speed
        self.vel_pub.publish(msg)

    def stop(self):
        for _ in range(8):
            self.command(0.0)
            rclpy.spin_once(self,timeout_sec=0.05)

    def drive(self,meters,speed=0.025):
        if not self.spin_until(lambda:self.odom is not None,10):
            raise RuntimeError('No wheel odometry for precision docking')
        p=self.odom.pose.pose.position
        start=(p.x,p.y)
        direction=1 if meters>=0 else -1
        target=abs(meters)
        start_time=time.monotonic()
        last_progress=start_time
        last_dist=0.0
        try:
            while rclpy.ok():
                rclpy.spin_once(self,timeout_sec=0.05)
                now=time.monotonic()
                if now-self.odom_time>1:
                    raise RuntimeError('Odometry stale: stopping docking')
                p=self.odom.pose.pose.position
                dist=math.hypot(p.x-start[0],p.y-start[1])
                if dist>=target-0.005:
                    return
                if dist>last_dist+0.005:
                    last_dist=dist
                    last_progress=now
                if now-last_progress>8 or now-start_time>max(30,5*target/speed):
                    raise RuntimeError('Docking motion stalled/timed out')
                self.command(direction*min(speed,max(0.005,(target-dist)*0.4)))
        finally:
            self.stop()

    def lift(self,height):
        cmd=Float64MultiArray()
        cmd.data=[height]
        deadline=time.monotonic()+3.0
        while time.monotonic()<deadline:
            self.lift_pub.publish(cmd)
            rclpy.spin_once(self,timeout_sec=0.05)
        self.get_logger().warning('Lift command sent; physical lift/cargo state NOT independently verified')

    def check_dock_alignment(self,wx,wy,wyaw):
        x,y,yaw=self.alignment.world_to_map(wx,wy,wyaw)
        rx,ry,ra=self.robot_pose()
        d=math.hypot(x-rx,y-ry)
        a=abs(angle_diff(yaw,ra))
        if d>self.args.dock_position_tolerance or a>math.radians(self.args.dock_yaw_tolerance):
            raise RuntimeError(f'Docking entry misaligned: {d:.3f}m / {math.degrees(a):.1f}deg')

    def run(self):
        self.init_alignment()
        pickup=STATIONS[self.args.pickup]
        drop=DROPS[self.args.dropoff]
        # Calibrated docking requires independent measurement of free corridor.
        # Robot approaches from east of SW station, facing west.
        if self.args.pickup!='sw':
            raise RuntimeError('Only SW pickup approach has a provisional geometry; NE not calibrated')
        px,py=pickup
        approach=(px+self.args.approach_offset,py,math.pi)
        if self.args.resume_unload:
            if not self.args.confirm_cargo:
                raise RuntimeError('Unloading requires --confirm-cargo')
            if self.args.pickup != 'sw' or self.args.dropoff != 'nw':
                raise RuntimeError('Resume-unload currently supports SW to NW only')
            self.get_logger().warning(
                'RESUME UNLOAD: navigate to NW marker, then lower lift'
            )
            dx,dy=drop
            self.navigate(dx,dy,math.pi,'FINAL DROPOFF')
            if self.args.preview:
                self.get_logger().info(
                    'Preview complete: no movement or lift commands'
                )
                return
            self.stop()
            self.lift(0.0)
            self.get_logger().warning(
                'Lift lowered; cargo release must be visually verified'
            )
            return

        if self.args.resume_loaded:
            if not self.args.confirm_cargo:
                raise RuntimeError('Resume requires --confirm-cargo')
            if self.args.preview:
                raise RuntimeError('Resume mode cannot use --preview')
            self.get_logger().warning(
                'RESUMING WITH CARGO: skipping pickup navigation, docking and lift'
            )
            self.drive(-self.args.dock_distance)
            dx,dy=drop
            self.navigate(
                dx+self.args.dropoff_offset,
                dy,
                math.pi,
                'LOADED DELIVERY APPROACH'
            )
            self.get_logger().warning(
                'STOP at delivery approach: unloading not calibrated'
            )
            return

        self.navigate(*approach,'PICKUP APPROACH')
        if self.args.preview:
            self.get_logger().info('Preview complete: no motion or lift commands were issued')
            return
        if not self.args.enable_docking:
            self.get_logger().warning('STOP at pickup approach: docking calibration not enabled')
            return
        self.check_dock_alignment(*approach)
        self.drive(self.args.dock_distance)
        self.lift(0.065)
        if not self.args.confirm_cargo:
            self.get_logger().warning('STOP: cargo pickup not verified. Do not navigate loaded.')
            return
        # Retreat independently of the forward docking distance.
        self.get_logger().info(
            f'RETREAT: reversing {self.args.retreat_distance:.2f} m'
        )
        self.drive(-self.args.retreat_distance)

        # Nav2 plans from the robot's NEW position after retreat.
        dx,dy=drop

        if self.args.auto_mission:
            self.get_logger().info(
                'AUTONOMOUS DELIVERY: planning from current position to NW'
            )
            self.navigate(dx,dy,math.pi,'AUTONOMOUS FINAL DROPOFF')
            self.stop()
            self.get_logger().info('Destination reached; lowering lift')
            self.lift(0.0)
            self.get_logger().warning(
                'Mission commands completed; visually verify cargo release'
            )
            return

        self.navigate(
            dx+self.args.dropoff_offset,
            dy,
            math.pi,
            'LOADED DELIVERY APPROACH'
        )
        if not self.args.enable_unload:
            self.get_logger().warning(
                'STOP at delivery approach: unloading not calibrated'
            )
            return
        raise RuntimeError(
            'Manual unloading remains blocked; use --auto-mission '
            'only after validating the delivery geometry'
        )


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pickup',choices=STATIONS,default='sw')
    p.add_argument('--dropoff',choices=DROPS,default='nw')
    p.add_argument('--preview',action='store_true',help='Plan and publish only; no robot motion')
    p.add_argument('--smoothing',action='store_true',default=True)
    p.add_argument('--no-smoothing',dest='smoothing',action='store_false')
    p.add_argument('--require-smoothing',action='store_true')
    p.add_argument('--nav-timeout',type=float,default=900)
    p.add_argument('--approach-offset',type=float,default=1.8)
    p.add_argument('--dock-distance',type=float,default=1.8)
    p.add_argument('--retreat-distance',type=float,default=1.0)
    p.add_argument(
        '--auto-mission',
        action='store_true',
        help='Automate SW pickup, 1 m retreat, NW delivery and unloading'
    )
    p.add_argument('--dropoff-offset',type=float,default=1.8)
    p.add_argument('--dock-position-tolerance',type=float,default=0.08)
    p.add_argument('--dock-yaw-tolerance',type=float,default=5.0)
    p.add_argument('--resume-unload',action='store_true',help='Navigate from current position to NW marker and lower lift')
    p.add_argument('--resume-loaded',action='store_true',help='Resume from docked position with cargo already lifted')
    p.add_argument('--enable-docking',action='store_true')
    p.add_argument('--confirm-cargo',action='store_true',help='Only use after independently verifying pickup')
    p.add_argument('--enable-unload',action='store_true')
    args=p.parse_args()
    if args.enable_unload and not args.confirm_cargo:
        p.error('--enable-unload requires --confirm-cargo')
    if (
        args.dock_distance <= 0
        or args.approach_offset <= 0
        or args.retreat_distance <= 0
    ):
        p.error('Distances must be positive')
    if args.auto_mission:
        if args.pickup != 'sw' or args.dropoff != 'nw':
            p.error('--auto-mission currently supports SW to NW only')
        if not args.enable_docking or not args.confirm_cargo:
            p.error(
                '--auto-mission requires --enable-docking '
                'and --confirm-cargo'
            )
    rclpy.init()
    node=Mission(args)
    try:
        node.run()
    except (RuntimeError,KeyboardInterrupt) as exc:
        node.get_logger().error(f'Mission stopped: {exc}')
        node.stop()
        sys.exit(2)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()
