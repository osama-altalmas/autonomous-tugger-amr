#!/usr/bin/env python3

import math
import rclpy

from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
from action_msgs.msg import GoalStatus


def yaw(q):
    return math.atan2(
        2 * (q.w * q.z + q.x * q.y),
        1 - 2 * (q.y * q.y + q.z * q.z)
    )


class PlanningTest(Node):

    def __init__(self):
        super().__init__("pickup_planning_test")

        self.client = ActionClient(
            self,
            ComputePathToPose,
            "/compute_path_to_pose"
        )

    def run(self):

        if not self.client.wait_for_server(timeout_sec=10):
            print("ERROR: Planner action unavailable.")
            return

        goal = ComputePathToPose.Goal()

        goal.planner_id = "GridBased"
        goal.use_start = False

        goal.goal = PoseStamped()
        goal.goal.header.frame_id = "map"
        goal.goal.header.stamp = self.get_clock().now().to_msg()

        goal.goal.pose.position.x = -15.70
        goal.goal.pose.position.y = -7.0
        goal.goal.pose.orientation.z = 1.0
        goal.goal.pose.orientation.w = 0.0

        print("Requesting path to southwest pickup...")
        print("No motion commands will be sent.")

        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        handle = future.result()

        if handle is None or not handle.accepted:
            print("ERROR: Planning goal rejected.")
            return

        result_future = handle.get_result_async()

        rclpy.spin_until_future_complete(
            self,
            result_future,
            timeout_sec=45
        )

        if not result_future.done():
            print("ERROR: Planning request timed out.")
            return

        response = result_future.result()

        if response.status != GoalStatus.STATUS_SUCCEEDED:
            print(f"PLANNING FAILED: status={response.status}")
            print(f"Result: {response.result}")
            return

        path = response.result.path
        poses = path.poses

        print(f"Path poses: {len(poses)}")

        if len(poses) < 2:
            print("ERROR: Path too short.")
            return

        forward = 0
        reverse = 0
        ambiguous = 0
        switches = 0

        previous_direction = None
        length = 0.0

        for a, b in zip(poses[:-1], poses[1:]):

            dx = b.pose.position.x - a.pose.position.x
            dy = b.pose.position.y - a.pose.position.y

            distance = math.hypot(dx, dy)

            if distance < 1e-5:
                continue

            length += distance

            heading = yaw(a.pose.orientation)

            projection = (
                dx * math.cos(heading) +
                dy * math.sin(heading)
            )

            if projection > 1e-4:
                direction = "FORWARD"
                forward += 1

            elif projection < -1e-4:
                direction = "REVERSE"
                reverse += 1

            else:
                direction = "AMBIGUOUS"
                ambiguous += 1
                continue

            if (
                previous_direction is not None
                and direction != previous_direction
            ):
                switches += 1

            previous_direction = direction

        first = poses[0].pose
        last = poses[-1].pose

        print()
        print("=== PATH RESULTS ===")
        print(f"Length: {length:.2f} m")

        print(
            f"Start: ({first.position.x:.3f}, "
            f"{first.position.y:.3f})"
        )

        print(
            f"End: ({last.position.x:.3f}, "
            f"{last.position.y:.3f})"
        )

        print(f"Forward segments: {forward}")
        print(f"Reverse segments: {reverse}")
        print(f"Ambiguous segments: {ambiguous}")
        print(f"Direction switches: {switches}")

        if forward > 0 and reverse == 0 and ambiguous == 0:
            print("RESULT: FORWARD-ONLY PATH")

        elif reverse > 0 and forward == 0 and ambiguous == 0:
            print("RESULT: REVERSE-ONLY PATH")

        else:
            print("RESULT: PATH REQUIRES FURTHER VALIDATION")


def main():

    rclpy.init()

    node = PlanningTest()

    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
