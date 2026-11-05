"""Isolated precision-command controller for camera docking.

This node is deliberately separate from the normal Nav2 and 4WS controllers.
It accepts only short-lived precision commands from the docking coordinator and
forwards them to the existing motion gate.  It never publishes wheel or
actuator commands directly.
"""

import math

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from std_msgs.msg import String


class DockingPrecisionController(Node):
    """Own the isolated camera-docking command boundary."""

    def __init__(self):
        super().__init__('tugger_docking_precision_controller')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
        self._latest = None
        self._latest_at = None
        self._owner = 'STOP'
        self._owner_at = None
        self._output = self.create_publisher(
            TwistStamped, '/tugger/precision_cmd_vel', 10)
        self.create_subscription(
            TwistStamped, '/tugger/docking/cmd_vel', self._on_command, 10)
        self.create_subscription(
            String, '/tugger/motion_owner', self._on_owner, 10)
        self.create_timer(0.05, self._tick)

    def _now(self):
        return self.get_clock().now()

    def _on_command(self, message):
        values = (message.twist.linear.x, message.twist.angular.z)
        if not all(math.isfinite(value) for value in values):
            self._latest = None
            self._latest_at = None
            return
        self._latest = message
        self._latest_at = self._now()

    def _on_owner(self, message):
        self._owner = message.data
        self._owner_at = self._now()

    def _tick(self):
        now = self._now()
        output = TwistStamped()
        output.header.stamp = now.to_msg()
        output.header.frame_id = 'base_footprint'

        owner_fresh = (
            self._owner_at is not None and
            0.0 <= (now - self._owner_at).nanoseconds / 1e9 <= 0.3)
        command_fresh = (
            self._latest is not None and self._latest_at is not None and
            0.0 <= (now - self._latest_at).nanoseconds / 1e9 <= 0.25)
        if owner_fresh and self._owner in ('PRECISION', 'DOCK_CAMERA') and command_fresh:
            output.twist.linear.x = self._latest.twist.linear.x
            output.twist.angular.z = self._latest.twist.angular.z
        self._output.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = DockingPrecisionController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
