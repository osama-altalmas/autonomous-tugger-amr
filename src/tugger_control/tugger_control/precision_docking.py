"""Geometry helpers for completing the final section of a saved dock route."""

import math


def final_approach_tail(poses, x, y):
    """Return the last saved-route section nearest to a premature stop.

    Nav2 may report the path complete before the independent docking check is
    satisfied.  Reusing the final saved poses preserves the clearance-checked
    approach and avoids asking the operator to press Dock a second time.
    """
    if len(poses) < 2:
        raise ValueError('A precision route needs at least two poses')
    nearest = min(
        range(len(poses)),
        key=lambda index: math.hypot(poses[index][0] - x, poses[index][1] - y))
    start = max(0, nearest - 1)
    tail = [list(pose) for pose in poses[start:]]
    if len(tail) < 2:
        tail = [list(poses[-2]), list(poses[-1])]
    return nearest, tail
