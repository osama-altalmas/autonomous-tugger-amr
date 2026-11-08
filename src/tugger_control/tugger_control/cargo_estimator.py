#!/usr/bin/env python3
"""Cargo state, cargo poses and collision scene without simulator truth (B4).

Publishes what the simulator used to provide, from the robot's own data
(see cargo_estimate.py):
  /cargo/station_N/state                      attached | detached
  /tugger/cargo/station_N/estimated_pose      cargo pose in map
  /tugger/collision_scene                     static site + cargo boxes
Inputs: the panel's attach/detach requests, the top QR reader, the lift
joint feedback and the localized robot pose (AMCL map -> base_footprint).
"""
import json
import math
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, String
from tf2_ros import Buffer, TransformListener

from tugger_control.cargo_estimate import CARGO_IDS, CargoEstimate, load_site, scene_payload


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class CargoEstimator(Node):
    def __init__(self):
        super().__init__('tugger_cargo_estimator')
        self.set_parameters([Parameter('use_sim_time', value=True)])
        world = Path(get_package_share_directory('tugger_gazebo')) / 'worlds/tugger_warehouse.sdf'
        self._static, self._shapes, poses, centres = load_site(world)
        self._estimate = CargoEstimate(poses, centres)
        self._lift = None
        self._tf = Buffer()
        self._listener = TransformListener(self._tf, self)
        self._state_pubs, self._pose_pubs = {}, {}
        for cargo_id in CARGO_IDS:
            self.create_subscription(Empty, f'/cargo/station_{cargo_id}/attach',
                                     lambda _m, c=cargo_id: self._request('attach', c), 10)
            self.create_subscription(Empty, f'/cargo/station_{cargo_id}/detach',
                                     lambda _m, c=cargo_id: self._request('detach', c), 10)
            self._state_pubs[cargo_id] = self.create_publisher(
                String, f'/cargo/station_{cargo_id}/state', 10)
            self._pose_pubs[cargo_id] = self.create_publisher(
                PoseStamped, f'/tugger/cargo/station_{cargo_id}/estimated_pose', 10)
        self.create_subscription(String, '/tugger/cargo/top_qr', self._on_qr, 10)
        self.create_subscription(JointState, '/joint_states', self._on_joints, 10)
        self._scene_pub = self.create_publisher(String, '/tugger/collision_scene', 10)
        self.create_timer(0.05, self._tick)
        self.get_logger().info('cargo estimator: state, poses and scene from lift, top QR and AMCL')

    def _now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _request(self, action, cargo_id):
        self._estimate.request(action, cargo_id)
        self.get_logger().info(f'{action} requested for cargo {cargo_id}')

    def _on_qr(self, msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        if data.get('id') in CARGO_IDS:
            self._estimate.qr_read(data['id'], self._now())

    def _on_joints(self, msg):
        if 'lift_joint' in msg.name:
            value = msg.position[msg.name.index('lift_joint')]
            if math.isfinite(value):
                self._lift = value

    def _robot_pose(self):
        try:
            t = self._tf.lookup_transform('map', 'base_footprint', rclpy.time.Time())
        except Exception:
            return None
        tr = t.transform.translation
        self._pose_stamp = t.header.stamp
        return (tr.x, tr.y, _yaw(t.transform.rotation))

    def _tick(self):
        now = self._now()
        pose = self._robot_pose()
        before = self._estimate.attached
        if pose is not None:
            self._estimate.update(pose, self._lift, now)
        if self._estimate.attached != before:
            self.get_logger().info(f'cargo {before if before is not None else self._estimate.attached} '
                                   f'{"attached" if self._estimate.attached is not None else "set down"}')
        # Stamp the cargo poses with the robot pose they were placed from, so
        # the motion gate compares them with the robot at the same instant.
        stamp = getattr(self, '_pose_stamp', None) or self.get_clock().now().to_msg()
        for cargo_id in CARGO_IDS:
            state = String()
            state.data = self._estimate.state(cargo_id)
            self._state_pubs[cargo_id].publish(state)
            x, y, z, yaw = self._estimate.poses[cargo_id]
            msg = PoseStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = 'map'
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = x, y, z
            msg.pose.orientation.z = math.sin(yaw / 2)
            msg.pose.orientation.w = math.cos(yaw / 2)
            self._pose_pubs[cargo_id].publish(msg)
        scene = String()
        scene.data = scene_payload(self._static, self._shapes, self._estimate.poses,
                                   stamp.sec + stamp.nanosec * 1e-9)
        self._scene_pub.publish(scene)


def main(args=None):
    rclpy.init(args=args)
    node = CargoEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
