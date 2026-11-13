"""Smooth back-out paths from a displaced robot onto its recorded approach.

A robot pushed sideways in a station is parallel to its approach but offset
from it. A straight line to a point further back starts with a heading kink a
car-like 4WS robot cannot follow from rest, so it turns in place under the
stand. Instead the path follows the reversed approach with the sideways
offset faded out by a quintic blend (zero slope and curvature at both ends):
it starts at the robot, parallel to the approach, and joins it without a kink.
Offline geometry only; no robot commands.
"""

import math

from tugger_control.steering_geometry import MAX_CURVATURE as FULL_LOCK_CURVATURE  # noqa: E402
# Leave room for steering slew and tracking error.
MAX_CURVATURE = 0.6 * FULL_LOCK_CURVATURE
BLEND_LENGTHS_M = (0.4, 0.6, 0.8, 1.0, 1.4, 1.8)
SPACING_M = 0.01


def _resample(points, spacing=SPACING_M):
    out = [tuple(points[0])]
    for a, b in zip(points, points[1:]):
        length = math.dist(a, b)
        steps = max(1, int(length / spacing))
        out.extend((a[0] + (b[0] - a[0]) * k / steps, a[1] + (b[1] - a[1]) * k / steps)
                   for k in range(1, steps + 1))
    return [p for i, p in enumerate(out) if i == 0 or math.dist(p, out[i - 1]) > 1e-6]


def _fade(u):
    """1 at u=0 to 0 at u=1; zero first and second derivative at both ends."""
    u = min(1.0, max(0.0, u))
    return 1.0 - u ** 3 * (10.0 - 15.0 * u + 6.0 * u * u)


def max_curvature(points):
    """Largest discrete curvature of a polyline (circumscribed circle)."""
    worst = 0.0
    for a, b, c in zip(points, points[2:], points[4:]):
        ab, bc, ca = math.dist(a, b), math.dist(b, c), math.dist(c, a)
        area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if ab * bc * ca > 1e-12:
            worst = max(worst, 2.0 * area2 / (ab * bc * ca))
    return worst


def backout_candidates(pose, reversed_approach, lengths=BLEND_LENGTHS_M,
                       curvature_limit=MAX_CURVATURE):
    """Yield (blend length, positions) from the robot onto the approach.

    reversed_approach runs from the point nearest the robot back to the gate.
    Candidates come shortest blend first and all respect curvature_limit.
    """
    route = _resample([(p[0], p[1]) for p in reversed_approach])
    if len(route) < 3:
        raise ValueError('approach too short to back out along')
    total = [0.0]
    for a, b in zip(route, route[1:]):
        total.append(total[-1] + math.dist(a, b))
    tangents = []
    for i in range(len(route)):
        a, b = route[max(0, i - 1)], route[min(len(route) - 1, i + 1)]
        norm = math.dist(a, b)
        tangents.append(((b[0] - a[0]) / norm, (b[1] - a[1]) / norm))
    tx, ty = tangents[0]
    offset = (pose[0] - route[0][0]) * -ty + (pose[1] - route[0][1]) * tx
    for length in lengths:
        if length > total[-1]:
            break
        positions = [(pose[0], pose[1])]
        for (x, y), (tx, ty), s in zip(route[1:], tangents[1:], total[1:]):
            shift = offset * _fade(s / length)
            positions.append((x - ty * shift, y + tx * shift))
        if max_curvature(positions) <= curvature_limit:
            yield length, positions
