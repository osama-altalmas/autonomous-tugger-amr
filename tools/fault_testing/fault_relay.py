#!/usr/bin/env python3
"""Fault testing only: pass the simulated scanners and IMU on, with faults.

The isolated simulation is started with sensor_faults:=true, so Gazebo's
laser scanners and IMU publish under /sim_raw/...; this relay republishes them
on the topics the robot reads. Without a fault the messages pass unchanged
(same stamp, same data). A fault is switched with a JSON command on
/fault/command, for example

  {"target": "front_left", "mode": "smudge", "start_deg": -20, "width_deg": 30}
  {"target": "rear_right", "mode": "silent"}
  {"target": "imu", "mode": "drift", "deg_per_hour": 15}
  {"target": "front_left", "mode": "none"}

Scanner faults (what each stands for on a real site):
  silent  the scanner stops sending (cable, power, crashed driver)
  smudge  dirt on the window: in a sector, some beams report a near return
          just outside the window (phantom object)
  blind   heavy dust or film on the window: in a sector, no returns at all
  ghost   reflective surfaces (glass, shrink-wrap, polished metal): now and
          then a few neighbouring beams report a point behind the real
          surface (a mirror image), and a small share of beams get no return
IMU fault:
  drift   gyro bias: the yaw rate is off by a constant and the yaw estimate
          drifts by the same rate (consumer MEMS: 2 to 15 deg per hour)

The robot code is not changed; this only stands in for imperfect sensors.
"""
import json
import math
import random

import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import String

SCANNERS = ('front_left', 'rear_right')
SMUDGE_SHARE = 0.35          # share of beams in the sector that see the dirt
SMUDGE_RANGE_M = (0.0, 0.12)  # beyond range_min: right outside the window
GHOST_SCAN_SHARE = 0.15      # share of scans with a mirror image
GHOST_BEAMS = (3, 8)
GHOST_BEHIND_M = (0.3, 2.0)
GHOST_DROPOUT = 0.02


def _angle_index(msg, deg):
    angle = math.radians(deg)
    return int(round((angle - msg.angle_min) / msg.angle_increment))


def apply_scan_fault(msg, fault, rng):
    """Return the ranges with the fault applied (a new list), or None to drop."""
    mode = fault.get('mode', 'none')
    if mode == 'silent':
        return None
    ranges = list(msg.ranges)
    n = len(ranges)
    if mode in ('smudge', 'blind'):
        first = _angle_index(msg, fault.get('start_deg', 0.0))
        count = max(1, int(round(math.radians(fault.get('width_deg', 30.0)) /
                                 abs(msg.angle_increment))))
        for k in range(count):
            i = first + k
            if not 0 <= i < n:
                continue
            if mode == 'blind':
                ranges[i] = math.inf
            elif rng.random() < fault.get('share', SMUDGE_SHARE):
                low, high = SMUDGE_RANGE_M
                ranges[i] = msg.range_min + rng.uniform(low, high)
    elif mode == 'ghost':
        if rng.random() < fault.get('scan_share', GHOST_SCAN_SHARE):
            width = rng.randint(*GHOST_BEAMS)
            start = rng.randrange(0, max(1, n - width))
            behind = rng.uniform(*GHOST_BEHIND_M)
            for i in range(start, start + width):
                if math.isfinite(ranges[i]):
                    ranges[i] = min(msg.range_max, ranges[i] + behind)
        for i in range(n):
            if rng.random() < fault.get('dropout', GHOST_DROPOUT):
                ranges[i] = math.inf
    return ranges


def drift_imu(msg, deg_per_hour, elapsed_s):
    """Add a constant yaw-rate bias and the yaw it accumulates."""
    bias = math.radians(deg_per_hour) / 3600.0
    msg.angular_velocity.z += bias
    half = bias * elapsed_s / 2.0
    q = msg.orientation
    c, s = math.cos(half), math.sin(half)
    # q_drift (rotation about z) * q
    msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = (
        c * q.w - s * q.z, c * q.x - s * q.y, c * q.y + s * q.x, c * q.z + s * q.w)
    return msg


class FaultRelay(Node):
    def __init__(self):
        super().__init__('tugger_fault_relay')
        self.faults = {name: {'mode': 'none'} for name in (*SCANNERS, 'imu')}
        self.rng = random.Random(7)
        self._imu_drift_start = None
        self._scan_pubs = {}
        for name in SCANNERS:
            self._scan_pubs[name] = self.create_publisher(LaserScan, f'/{name}/scan', 10)
            self.create_subscription(LaserScan, f'/sim_raw/{name}/scan',
                                     lambda msg, n=name: self._scan(n, msg),
                                     qos_profile_sensor_data)
        self._imu_pub = self.create_publisher(Imu, '/imu/data', 50)
        self.create_subscription(Imu, '/sim_raw/imu/data', self._imu, qos_profile_sensor_data)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._state_pub = self.create_publisher(String, '/fault/state', latched)
        self.create_subscription(String, '/fault/command', self._command, 10)
        self._publish_state()
        self.get_logger().info('fault relay ready: scanners and IMU pass through')

    def _publish_state(self):
        self._state_pub.publish(String(data=json.dumps(self.faults)))

    def _command(self, msg):
        try:
            command = json.loads(msg.data)
            target = command.pop('target')
            if target not in self.faults:
                raise ValueError(f'unknown target {target}')
        except (ValueError, KeyError) as error:
            self.get_logger().error(f'bad fault command {msg.data!r}: {error}')
            return
        self.faults[target] = command
        if target == 'imu':
            self._imu_drift_start = None
        self.get_logger().info(f'fault {target}: {command}')
        self._publish_state()

    def _scan(self, name, msg):
        fault = self.faults[name]
        if fault.get('mode', 'none') == 'none':
            self._scan_pubs[name].publish(msg)
            return
        ranges = apply_scan_fault(msg, fault, self.rng)
        if ranges is None:
            return
        msg.ranges = ranges
        self._scan_pubs[name].publish(msg)

    def _imu(self, msg):
        fault = self.faults['imu']
        if fault.get('mode') == 'drift':
            now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if self._imu_drift_start is None:
                self._imu_drift_start = now
            msg = drift_imu(msg, fault.get('deg_per_hour', 15.0), now - self._imu_drift_start)
        self._imu_pub.publish(msg)


def main():
    rclpy.init()
    node = FaultRelay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
