#!/usr/bin/env python3
"""
AMR V1 — Gazebo saved-route follower.

Reads a previously validated static route and follows it
through the existing 4WS /cmd_vel TwistStamped interface.

Default: PREFLIGHT ONLY. No commands published.

Motion requires BOTH:
  --enable-motion
  --gazebo-only

This is an initial simulation controller, NOT a certified
industrial safety controller. No dynamic obstacle stopping,
person detection, cargo docking, or lifting is implemented.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import yaml

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time

from geometry_msgs.msg import TwistStamped
from tf2_ros import Buffer, TransformListener, TransformException

from build_network import (
    load_occupancy,
    validate_trajectory,
)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def yaw_of(q):
    return math.atan2(
        2 * (q.w * q.z + q.x * q.y),
        1 - 2 * (q.y * q.y + q.z * q.z)
    )


def current_map_hash(path):
    path = Path(path)
    info = yaml.safe_load(path.read_text())
    image = path.parent / info["image"]

    return hashlib.sha256(
        path.read_bytes() + image.read_bytes()
    ).hexdigest()


def load_and_validate(args):
    db_file = Path(args.database)
    db = json.loads(db_file.read_text())

    fingerprint = current_map_hash(args.map)

    if db["map_sha256"] != fingerprint:
        raise ValueError(
            "Current occupancy map does not match the "
            "saved-route database"
        )

    key = f"{args.source}__TO__{args.target}"
    entry = db["routes"].get(key)

    if not entry:
        raise ValueError("Unknown saved route: " + key)

    if entry["status"] != "VALIDATED_STATIC_CANDIDATE":
        raise ValueError("Route is not validated: " + key)

    route_file = db_file.parent / entry["route_file"]
    route = json.loads(route_file.read_text())

    if (
        route["source"] != args.source
        or route["target"] != args.target
        or route["map_sha256"] != fingerprint
    ):
        raise ValueError("Saved route identity/hash mismatch")

    poses = route["poses"]

    if len(poses) < 2:
        raise ValueError("Route has insufficient poses")

    if any(
        len(p) != 3
        or not all(math.isfinite(float(v)) for v in p)
        for p in poses
    ):
        raise ValueError("Invalid route pose")

    direction = route.get("motion_direction", "FORWARD")

    if direction not in ("FORWARD", "REVERSE"):
        raise ValueError("Invalid motion direction")

    map_spec, blocked = load_occupancy(args.map)

    collision = validate_trajectory(
        poses, map_spec, blocked
    )

    if not collision["valid"]:
        raise ValueError(
            "Route footprint failed static collision validation"
        )

    # Verify body orientation versus actual displacement.
    # This is essential for correct reverse driving.
    worst = 0.0

    for a, b in zip(poses, poses[1:]):
        dx = b[0] - a[0]
        dy = b[1] - a[1]

        if math.hypot(dx, dy) < 1e-7:
            continue

        travel = math.atan2(dy, dx)

        expected = a[2] + (
            math.pi if direction == "REVERSE" else 0.0
        )

        worst = max(worst, abs(wrap(travel - expected)))

    if worst > math.radians(12):
        raise ValueError(
            "Route body headings disagree with motion direction"
        )

    return route, direction, poses


class SavedRouteFollower(Node):

    def __init__(self, args, poses, direction):
        super().__init__("amr_v1_saved_path_follower")

        # Gazebo, TF, EKF, and this follower must all
        # use the same ROS simulation clock.
        results = self.set_parameters([
            Parameter(
                "use_sim_time",
                Parameter.Type.BOOL,
                True
            )
        ])

        if not all(result.successful for result in results):
            raise RuntimeError(
                "Cannot enable Gazebo simulation time"
            )

        self.args = args
        self.poses = poses
        self.direction = direction
        self.arc = self.route_distances()
        self.previous_command_speed = 0.0
        self.motion_enabled = (
            args.enable_motion and args.gazebo_only
        )

        self.buffer = Buffer()
        self.listener = TransformListener(
            self.buffer, self
        )

        self.publisher = self.create_publisher(
            TwistStamped, "/cmd_vel", 10
        )

        self.phase = "PREFLIGHT"
        self.tf_startup_deadline = time.monotonic() + 10.0
        self.tf_wait_logged = False
        self.index = 0
        self.start_wall = None
        self.last_pose_wall = None
        self.last_pose = None
        self.last_log = 0.0
        self.failed_reason = None
        self.finished = False

        self.timer = self.create_timer(
            0.05, self.update
        )

        self.get_logger().info(
            f"Loaded {len(poses)} poses, "
            f"direction={direction}; "
            f"motion_enabled={self.motion_enabled}"
        )

    def command(self, v=0.0, omega=0.0):
        if not self.motion_enabled:
            return

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(omega)

        self.publisher.publish(msg)

    def stop(self, reason, failed=False):
        if self.finished:
            return

        self.command(0.0, 0.0)

        self.finished = True
        self.failed_reason = reason if failed else None

        if failed:
            self.get_logger().error(
                "STOPPED: " + reason
            )
        else:
            self.get_logger().info(
                "STOPPED: " + reason
            )

    def robot_pose(self):
        transform = self.buffer.lookup_transform(
            "map", "base_footprint", Time()
        )

        stamp = transform.header.stamp
        stamp_ns = (
            stamp.sec * 1_000_000_000
            + stamp.nanosec
        )

        now_ns = self.get_clock().now().nanoseconds

        age = (now_ns - stamp_ns) / 1e9

        if age < -0.3 or age > self.args.tf_timeout:
            raise ValueError(
                f"Stale robot localization ({age:.2f}s)"
            )

        tr = transform.transform

        return (
            float(tr.translation.x),
            float(tr.translation.y),
            yaw_of(tr.rotation)
        )

    def closest_index(self, x, y):
        # Permit a small backward search to tolerate noise.
        start = max(0, self.index - 12)

        finish = min(
            len(self.poses),
            self.index + 250
        )

        nearest = min(
            range(start, finish),
            key=lambda i: (
                (self.poses[i][0] - x) ** 2
                + (self.poses[i][1] - y) ** 2
            )
        )

        self.index = max(self.index, nearest)

        error = distance(
            (x, y), self.poses[nearest]
        )

        return nearest, error

    def lookahead_point(self, index, lookahead):
        remaining = lookahead

        for i in range(index, len(self.poses) - 1):
            a = self.poses[i]
            b = self.poses[i + 1]

            length = distance(a, b)

            if length >= remaining:
                ratio = remaining / max(length, 1e-8)

                return (
                    a[0] + ratio * (b[0] - a[0]),
                    a[1] + ratio * (b[1] - a[1])
                )

            remaining -= length

        return tuple(self.poses[-1][:2])

    def route_distances(self):
        """Cumulative distance along the saved path."""
        distances = [0.0]

        for a, b in zip(self.poses, self.poses[1:]):
            distances.append(
                distances[-1] + distance(a, b)
            )

        return distances

    def bend_speed_limit(self, index, progress, braking):
        """
        Limit speed for upcoming path curvature.

        We scan ahead and calculate the speed that can
        be reached before each curve using the configured
        braking deceleration.
        """
        limit = self.args.speed
        max_lat = self.args.max_lateral_accel

        for j in range(index, len(self.poses) - 1):
            ahead = self.arc[j] - progress

            if ahead > 6.0:
                break

            a = self.poses[j]
            b = self.poses[j + 1]

            ds = max(distance(a, b), 1e-6)

            da = abs(wrap(b[2] - a[2]))
            kappa = da / ds

            if kappa < 1e-4:
                continue

            curve_speed = math.sqrt(
                max_lat / kappa
            )

            braking_distance = max(
                0.0,
                ahead - 0.30
            )

            reachable = math.sqrt(
                curve_speed ** 2
                + 2.0 * braking * braking_distance
            )

            limit = min(limit, reachable)

        return limit

    def update(self):
        if self.finished:
            return

        try:
            x, y, yaw = self.robot_pose()

        except (TransformException, ValueError) as exc:
            if self.phase == "PREFLIGHT":
                if time.monotonic() < self.tf_startup_deadline:
                    if not self.tf_wait_logged:
                        self.get_logger().info(
                            "Waiting for map -> base_footprint TF "
                            "to initialize (maximum 10 seconds)."
                        )
                        self.tf_wait_logged = True
                    return

                self.stop(
                    f"Localization preflight failed after "
                    f"10 seconds: {exc}",
                    failed=True
                )
            else:
                self.stop(
                    f"Localization lost: {exc}",
                    failed=True
                )

            return

        now = time.monotonic()

        if self.phase == "PREFLIGHT":
            if self.args.resume:
                # Find the closest point on the SAVED path.
                # This is allowed only during an explicit
                # operator-requested resume.
                nearest = min(
                    range(len(self.poses)),
                    key=lambda i: (
                        (self.poses[i][0] - x) ** 2
                        + (self.poses[i][1] - y) ** 2
                    )
                )

                self.index = nearest
                start = self.poses[nearest]

                self.get_logger().info(
                    f"RESUME REQUESTED: nearest saved "
                    f"path index={nearest}/{len(self.poses)-1}"
                )
            else:
                start = self.poses[0]

            xy_error = distance(
                (x, y), start
            )

            yaw_error = abs(wrap(
                yaw - start[2]
            ))

            self.get_logger().info(
                f"START CHECK: current=({x:.3f},{y:.3f},"
                f"{math.degrees(yaw):.1f}deg) "
                f"route=({start[0]:.3f},{start[1]:.3f},"
                f"{math.degrees(start[2]):.1f}deg) "
                f"xy_error={xy_error:.3f}m "
                f"yaw_error={math.degrees(yaw_error):.1f}deg"
            )

            if xy_error > self.args.start_tolerance:
                self.stop(
                    "Robot is not at the saved route start",
                    failed=True
                )
                return

            if yaw_error > math.radians(
                self.args.start_yaw_deg
            ):
                self.stop(
                    "Robot heading does not match route start",
                    failed=True
                )
                return

            if not self.motion_enabled:
                self.stop(
                    "PREFLIGHT PASSED — motion remains disabled"
                )
                return

            self.start_wall = now
            self.phase = "FOLLOWING"
            self.get_logger().warning(
                "GAZEBO MOTION ENABLED. "
                "Press Ctrl+C to stop."
            )

        if (
            self.start_wall is not None
            and now - self.start_wall > self.args.max_seconds
        ):
            self.stop(
                "Maximum driving time reached",
                failed=True
            )
            return

        index, tracking_error = self.closest_index(
            x, y
        )

        if tracking_error > self.args.max_tracking_error:
            self.stop(
                f"Path deviation {tracking_error:.3f}m "
                "exceeded limit",
                failed=True
            )
            return

        final = self.poses[-1]

        remaining = distance(
            (x, y), final
        )

        heading_error = abs(wrap(
            yaw - final[2]
        ))

        if (
            remaining < self.args.goal_tolerance
            and heading_error < math.radians(
                self.args.goal_yaw_deg
            )
        ):
            self.stop("GOAL REACHED")
            return

        if (
            index >= len(self.poses) - 2
            and remaining > self.args.max_tracking_error
        ):
            self.stop(
                "Overshot final path position",
                failed=True
            )
            return

        # Distance along the saved trajectory, not the
        # straight-line distance to the destination.
        progress = self.arc[index]
        route_remaining = max(
            0.0,
            self.arc[-1] - progress
        )

        # Increase lookahead when speed is high.
        adaptive_lookahead = min(
            2.5,
            max(
                self.args.lookahead,
                0.65 + 0.9 * abs(
                    self.previous_command_speed
                )
            )
        )

        target = self.lookahead_point(
            index,
            adaptive_lookahead
        )

        alpha = wrap(
            math.atan2(
                target[1] - y,
                target[0] - x
            ) - yaw
        )

        if self.direction == "REVERSE":
            alpha = wrap(alpha - math.pi)

        lookahead_distance = max(
            adaptive_lookahead,
            distance((x, y), target)
        )

        curvature = (
            2.0 * math.sin(alpha)
            / lookahead_distance
        )

        max_kappa = (
            2.0 * math.tan(self.args.max_steering)
            / self.args.wheelbase
        )

        curvature = max(
            -max_kappa,
            min(max_kappa, curvature)
        )

        # Respect instantaneous cornering acceleration.
        tracking_speed_limit = math.sqrt(
            self.args.max_lateral_accel
            / max(abs(curvature), 1e-5)
        )

        # Slow down before upcoming bends.
        bend_speed_limit = self.bend_speed_limit(
            index,
            progress,
            self.args.braking_deceleration
        )

        # Stop-distance constraint based on route distance.
        # Hold back a small margin for final positioning.
        stopping_distance = max(
            0.0,
            route_remaining - getattr(
                self.args, "braking_margin", 0.08
            )
        )

        braking_speed_limit = math.sqrt(
            2.0
            * self.args.braking_deceleration
            * stopping_distance
        )

        speed = min(
            self.args.speed,
            tracking_speed_limit,
            bend_speed_limit,
            braking_speed_limit
        )

        # Low-speed precision approach near the goal.
        if route_remaining < 0.8:
            speed = min(
                speed,
                max(0.025, 0.25 * route_remaining)
            )

        if self.direction == "REVERSE":
            speed *= -1.0

        omega = speed * curvature

        self.previous_command_speed = speed
        self.command(speed, omega)

        if now - self.last_log > 1.0:
            self.get_logger().info(
                f"tracking index={index}/{len(self.poses)} "
                f"error={tracking_error:.3f}m "
                f"goal_distance={remaining:.2f}m "
                f"path_remaining={route_remaining:.2f}m "
                f"lookahead={adaptive_lookahead:.2f}m "
                f"v={speed:.3f}m/s "
                f"omega={omega:.3f}rad/s"
            )

            self.last_log = now


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--database", required=True)
    ap.add_argument("--map", required=True)

    ap.add_argument("--from", dest="source", required=True)
    ap.add_argument("--to", dest="target", required=True)

    ap.add_argument("--enable-motion", action="store_true")
    ap.add_argument("--gazebo-only", action="store_true")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Start at the nearest saved path pose "
             "after checking position and heading"
    )

    ap.add_argument("--speed", type=float, default=0.08)
    ap.add_argument("--lookahead", type=float, default=0.65)

    ap.add_argument(
        "--max-tracking-error",
        type=float,
        default=0.35
    )

    ap.add_argument(
        "--start-tolerance",
        type=float,
        default=0.20
    )

    ap.add_argument(
        "--start-yaw-deg",
        type=float,
        default=8.0
    )

    ap.add_argument(
        "--goal-tolerance",
        type=float,
        default=0.12
    )

    ap.add_argument(
        "--goal-yaw-deg",
        type=float,
        default=10.0
    )

    ap.add_argument(
        "--tf-timeout",
        type=float,
        default=0.40
    )

    ap.add_argument(
        "--max-seconds",
        type=float,
        default=600.0
    )

    ap.add_argument(
        "--wheelbase",
        type=float,
        default=0.58
    )

    ap.add_argument(
        "--max-steering",
        type=float,
        default=math.pi / 4
    )

    ap.add_argument(
        "--max-lateral-accel",
        type=float,
        default=0.08
    )

    ap.add_argument(
        "--braking-deceleration",
        type=float,
        default=0.35
    )

    args = ap.parse_args()

    if args.enable_motion != args.gazebo_only:
        ap.error(
            "Motion requires BOTH --enable-motion "
            "and --gazebo-only"
        )

    if not (
        0 < args.speed <= 2.5
        and 0.2 <= args.lookahead <= 1.5
        and 0 < args.max_tracking_error <= 0.5
        and 0 < args.wheelbase
        and 0 < args.max_lateral_accel
        and 0 < args.max_seconds
    ):
        ap.error("Invalid or excessive motion parameters")

    route, direction, poses = load_and_validate(args)

    print()
    print("========== AMR V1 PATH FOLLOWER ==========")
    print("Route:", args.source, "->", args.target)
    print("Length:", route["length_m"], "m")
    print("Poses:", len(poses))
    print("Direction:", direction)
    print("Static collision check: PASSED")
    print("Maximum speed:", args.speed, "m/s")
    print(
        "Mode:",
        "GAZEBO MOTION" if args.enable_motion
        else "PREFLIGHT ONLY"
    )
    print("Dynamic obstacle avoidance: NOT IMPLEMENTED")
    print("==========================================")

    rclpy.init()

    node = SavedRouteFollower(
        args, poses, direction
    )

    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(
                node,
                timeout_sec=0.10
            )

    except KeyboardInterrupt:
        node.stop(
            "Operator requested stop",
            failed=True
        )

    finally:
        # Repeated zero commands give the downstream
        # controller an opportunity to receive STOP.
        if node.motion_enabled:
            for _ in range(10):
                node.command(0.0, 0.0)
                rclpy.spin_once(
                    node,
                    timeout_sec=0.05
                )

        failed = node.failed_reason is not None

        node.destroy_node()
        rclpy.shutdown()

        if failed:
            sys.exit(2)


if __name__ == "__main__":
    main()
