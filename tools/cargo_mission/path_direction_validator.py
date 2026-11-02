#!/usr/bin/env python3
"""Validate that a Nav2 path uses exactly one driving direction."""

import math


def yaw_of(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def validate_path_direction(path, min_segment=1e-5,
                            min_projection_ratio=0.05):
    """
    Return 'forward' or 'reverse'.

    Raise ValueError if the path is degenerate, has ambiguous
    segments, or contains a direction change.
    """
    poses = path.poses

    if len(poses) < 2:
        raise ValueError("Path contains fewer than two poses")

    directions = set()
    checked = 0

    for a, b in zip(poses[:-1], poses[1:]):
        dx = b.pose.position.x - a.pose.position.x
        dy = b.pose.position.y - a.pose.position.y
        distance = math.hypot(dx, dy)

        if distance < min_segment:
            continue

        heading = yaw_of(a.pose.orientation)

        projection = (
            dx * math.cos(heading) +
            dy * math.sin(heading)
        )

        ratio = projection / distance

        if ratio > min_projection_ratio:
            directions.add("forward")
        elif ratio < -min_projection_ratio:
            directions.add("reverse")
        else:
            raise ValueError(
                "Path contains a segment with ambiguous direction"
            )

        checked += 1

        if len(directions) > 1:
            raise ValueError(
                "Path contains both forward and reverse segments"
            )

    if checked == 0 or len(directions) != 1:
        raise ValueError("Path has no valid directional segments")

    return directions.pop()


if __name__ == "__main__":
    print("Path direction validator created successfully.")
