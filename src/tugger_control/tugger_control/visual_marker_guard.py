"""Camera and map-TF agreement checks for precision alignment."""

import math


class FixedMarkerGuard:
    """Keep the stationary target fixed in map coordinates."""

    def __init__(self, robot_pose, marker_relative, *,
                 agreement=0.012,
                 minimum_radius=0.060,
                 extra_radius=0.035,
                 max_radius=0.090,
                 max_yaw=math.radians(10)):

        values = (*robot_pose, *marker_relative)

        if (
            len(robot_pose) != 3
            or len(marker_relative) != 2
            or not all(math.isfinite(v) for v in values)
        ):
            raise ValueError("Finite map and camera poses required")

        x, y, yaw = robot_pose
        u, v = marker_relative

        c = math.cos(yaw)
        s = math.sin(yaw)

        self.marker = (
            x + c * u - s * v,
            y + s * u + c * v,
        )

        self.radius = min(
            max_radius,
            max(
                minimum_radius,
                math.hypot(u, v) + extra_radius,
            ),
        )

        self.agreement = agreement
        self.max_yaw = max_yaw

    def check(self, robot_pose, marker_relative, target_heading):
        """Return OK, HOLD, or ABORT with an explanation."""

        try:
            x, y, yaw = robot_pose
            u, v = marker_relative

            if not all(
                math.isfinite(t)
                for t in (x, y, yaw, u, v, target_heading)
            ):
                return "ABORT", "nonfinite map/vision pose"

        except (ValueError, TypeError):
            return "ABORT", "missing map/vision pose"

        distance = math.hypot(
            x - self.marker[0],
            y - self.marker[1],
        )

        if distance > self.radius:
            return (
                "ABORT",
                f"capture boundary exceeded: "
                f"{distance:.3f} m > {self.radius:.3f} m",
            )

        yaw_error = math.atan2(
            math.sin(yaw - target_heading),
            math.cos(yaw - target_heading),
        )

        if abs(yaw_error) > self.max_yaw:
            return (
                "ABORT",
                "heading departed the camera capture envelope",
            )

        c = math.cos(yaw)
        s = math.sin(yaw)

        seen = (
            x + c * u - s * v,
            y + s * u + c * v,
        )

        discrepancy = math.hypot(
            seen[0] - self.marker[0],
            seen[1] - self.marker[1],
        )

        if discrepancy > self.agreement:
            return (
                "HOLD",
                f"camera and TF disagree on fixed marker "
                f"by {discrepancy:.3f} m",
            )

        return "OK", ""
