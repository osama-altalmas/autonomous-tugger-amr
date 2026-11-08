#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry

from tugger_control.turn_in_place import pivot_odometry


class FourWSOdometry(Node):

    def __init__(self):
        super().__init__('four_ws_odometry')

        self.declare_parameter('wheelbase', 0.58)
        self.declare_parameter('track_width', 0.44)
        self.declare_parameter('wheel_radius', 0.055)

        self.wheelbase = float(
            self.get_parameter('wheelbase').value
        )
        self.track_width = float(
            self.get_parameter('track_width').value
        )
        self.wheel_radius = float(
            self.get_parameter('wheel_radius').value
        )

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.last_time = None

        self.odom_pub = self.create_publisher(
            Odometry,
            '/wheel/odom',
            10
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            20
        )

        self.get_logger().info(
            '4WS linked-axle wheel odometry started '
            f'(L={self.wheelbase:.3f} m, '
            f'W={self.track_width:.3f} m, '
            f'R={self.wheel_radius:.3f} m)'
        )

    def joint_state_callback(self, msg):

        joint_index = {
            name: i for i, name in enumerate(msg.name)
        }

        required_joints = [
            'rear_left_steering_joint',
            'rear_right_steering_joint',
            'rear_left_wheel_joint',
            'rear_right_wheel_joint',
        ]

        if not all(name in joint_index for name in required_joints):
            return

        rl_steer_index = joint_index['rear_left_steering_joint']
        rr_steer_index = joint_index['rear_right_steering_joint']
        rl_wheel_index = joint_index['rear_left_wheel_joint']
        rr_wheel_index = joint_index['rear_right_wheel_joint']

        if len(msg.position) <= max(
            rl_steer_index,
            rr_steer_index
        ):
            return

        if len(msg.velocity) <= max(
            rl_wheel_index,
            rr_wheel_index
        ):
            return

        delta_rl = msg.position[rl_steer_index]
        delta_rr = msg.position[rr_steer_index]

        # Mechanical linkage means these should be equal.
        # Average measured positions to reduce small simulation/controller
        # tracking differences.
        rear_angle = 0.5 * (
            delta_rl + delta_rr
        )

        omega_rl_wheel = msg.velocity[rl_wheel_index]
        omega_rr_wheel = msg.velocity[rr_wheel_index]

        speed_rl = (
            omega_rl_wheel *
            self.wheel_radius
        )

        speed_rr = (
            omega_rr_wheel *
            self.wheel_radius
        )

        # Recover body longitudinal velocity from both rear wheels.
        #
        # With common rear steering angle:
        #
        # s_rl*cos(delta_r) = v - wz*W/2
        # s_rr*cos(delta_r) = v + wz*W/2
        #
        # Averaging eliminates the yaw-rate contribution.
        linear_velocity = 0.5 * (
            speed_rl + speed_rr
        ) * math.cos(rear_angle)

        # Symmetric counter-phase 4WS:
        #
        # delta_front = -delta_rear
        #
        # wz / v = 2*tan(delta_front)/L
        #        = -2*tan(delta_rear)/L
        lateral_velocity = 0.0
        front = self._front_wheels(msg, joint_index)
        # Car-like counter-phase steering has front = -rear. Anything else
        # (crab: all four at one angle) is not the linked-axle model.
        crab_like = (front is not None and
                     abs(0.5 * (front[0] + front[1]) + rear_angle) > 0.05)
        if abs(delta_rl - delta_rr) > 0.05 or crab_like:
            # Rear wheels not linked (turning in place, crab, or changing
            # between patterns): fit the rigid-body motion to every measured
            # wheel instead of the linked-axle model.
            half_l, half_w = 0.5 * self.wheelbase, 0.5 * self.track_width
            steering = [delta_rl, delta_rr]
            speeds = [omega_rl_wheel, omega_rr_wheel]
            positions = [(-half_l, half_w), (-half_l, -half_w)]
            if front is not None:
                steering += [front[0], front[1]]
                speeds += [front[2], front[3]]
                positions += [(half_l, half_w), (half_l, -half_w)]
            linear_velocity, lateral_velocity, angular_velocity = pivot_odometry(
                steering, speeds, wheel_positions=positions,
                wheel_radius=self.wheel_radius)
        elif abs(self.wheelbase) > 1e-9:
            angular_velocity = (
                -2.0 *
                linear_velocity *
                math.tan(rear_angle) /
                self.wheelbase
            )
        else:
            angular_velocity = 0.0

        current_time = self.get_clock().now()

        if self.last_time is None:
            self.last_time = current_time
            return

        dt = (
            current_time -
            self.last_time
        ).nanoseconds / 1e9

        self.last_time = current_time

        if dt <= 0.0 or dt > 0.5:
            return

        delta_yaw = angular_velocity * dt

        yaw_mid = (
            self.yaw +
            0.5 * delta_yaw
        )

        self.x += (
            linear_velocity * math.cos(yaw_mid) -
            lateral_velocity * math.sin(yaw_mid)
        ) * dt

        self.y += (
            linear_velocity * math.sin(yaw_mid) +
            lateral_velocity * math.cos(yaw_mid)
        ) * dt

        self.yaw += delta_yaw

        self.yaw = math.atan2(
            math.sin(self.yaw),
            math.cos(self.yaw)
        )

        self.publish_odometry(
            current_time,
            linear_velocity,
            angular_velocity,
            lateral_velocity
        )

    @staticmethod
    def _front_wheels(msg, joint_index):
        """(fl angle, fr angle, fl speed, fr speed) or None if not measured."""
        names = ('front_left_steering_joint', 'front_right_steering_joint',
                 'front_left_wheel_joint', 'front_right_wheel_joint')
        if not all(name in joint_index for name in names):
            return None
        fl_s, fr_s, fl_w, fr_w = (joint_index[name] for name in names)
        if (len(msg.position) <= max(fl_s, fr_s) or
                len(msg.velocity) <= max(fl_w, fr_w)):
            return None
        values = (msg.position[fl_s], msg.position[fr_s],
                  msg.velocity[fl_w], msg.velocity[fr_w])
        return values if all(math.isfinite(v) for v in values) else None

    def publish_odometry(
        self,
        current_time,
        linear_velocity,
        angular_velocity,
        lateral_velocity=0.0
    ):

        half_yaw = 0.5 * self.yaw

        odom = Odometry()

        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_footprint'

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0

        odom.pose.pose.orientation.z = math.sin(half_yaw)
        odom.pose.pose.orientation.w = math.cos(half_yaw)

        odom.twist.twist.linear.x = linear_velocity
        odom.twist.twist.linear.y = lateral_velocity
        odom.twist.twist.angular.z = angular_velocity

        odom.pose.covariance[0] = 0.02
        odom.pose.covariance[7] = 0.02
        odom.pose.covariance[35] = 0.05

        odom.twist.covariance[0] = 0.01
        odom.twist.covariance[7] = 0.02
        odom.twist.covariance[35] = 0.03

        self.odom_pub.publish(odom)



def main(args=None):
    rclpy.init(args=args)

    node = FourWSOdometry()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
