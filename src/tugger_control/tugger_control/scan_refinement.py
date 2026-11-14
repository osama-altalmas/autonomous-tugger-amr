"""Scan-to-map refinement of the laser localization (pure functions, no ROS).

AMCL's particle filter settles 2-3 cm from the true pose (front-to-back 2.1 cm
rms, sideways 1.6 cm rms over 22 Gazebo runs, 2026-11-14), while a single
scan of the two corner scanners fits the map to about 0.4 cm. The particle
cloud's mean is never sharper than the cloud. Precise AMRs therefore use the
particle filter for the rough pose and a scan-to-map match for the last
centimetres; this module is that match.

The unknown is the map -> odom transform. Scan points are first expressed in
the odom frame through each scan's own odometry pose (so two scanners firing
40 ms apart, and the robot's motion between them, cost nothing), then the
transform is solved so the points lie on the map's walls: Gauss-Newton on a
smooth distance field, robust (Huber) weights, far points (unmapped objects,
people, cargo) dropped.

Degenerate directions keep the prior. In a long featureless corridor the
scan cannot tell front from back; the fit would wander along it on noise.
The problem is solved in the eigen-directions of its information matrix and
only well-constrained directions are moved (solution remapping, Zhang et al.,
"On degeneracy of optimization-based state estimation problems", ICRA 2016);
the others stay where AMCL put them.

The result is used only when it is clearly good (enough points on walls, small
residual, converged) and close to the prior; otherwise the caller keeps
AMCL's pose. A bad match can therefore never move the robot far.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

# A point farther than this from any mapped wall is not on the map (cargo,
# a person, a stand) and is left out of the fit.
FAR_M = 0.30
# Huber threshold: residuals beyond it count linearly, not squared.
HUBER_M = 0.02
# A point within this distance of a wall after the fit counts as on the wall.
INLIER_M = 0.03
# Rotation is scaled by this length to compare it with translation (1 rad at
# 1 m equals 1 m), so the information matrix has one unit.
ROTATION_LENGTH_M = 1.0
# A direction is constrained when the scan holds at least this much
# information along it, in "points squarely on a wall": 30 such points with
# 1 cm noise fix it to about 2 mm.
MIN_INFORMATION = 30.0
# Acceptance limits.
MIN_INLIERS = 150
MIN_INLIER_FRACTION = 0.6
# Mean gap between scan points and walls after a correct fit: about 1.2-1.4
# cm in Gazebo (1 cm range noise plus the 2 cm map cells). A limit of 1.2 cm
# refused 61% of correct fits on a full cycle (2026-11-14); a wrong match
# leaves far more than 2 cm, and real scanners are noisier than 1 cm.
MAX_RMS_M = 0.020
# Largest correction of AMCL accepted. AMCL is off by up to ~10 cm and ~1.5
# degrees at times (offline replay of recorded runs: the refined pose was
# right in all of those); a wrong match shows in the residual and the share
# of points on walls, not only in its size.
MAX_SHIFT_M = 0.15
MAX_TURN_RAD = math.radians(3.0)
MAX_ITERATIONS = 15


class DistanceField:
    """Signed distance (m) to the nearest wall surface, smooth between cells.

    Positive in free space, negative inside walls, zero on a wall's surface
    (a cell edge): distances between cell centres, less half a cell. Inside
    walls it keeps growing with depth, so a point the prior put a few cm into
    a wall is pulled back out; an unsigned field is flat there, and those
    points (the very ones that fix front-to-back on the wall ahead) gave no
    information at all. Bilinear interpolation keeps it continuous, which a
    Gauss-Newton fit needs (a nearest-cell lookup is a staircase).
    """

    def __init__(self, occupied, resolution, origin, max_distance=1.0):
        occupied = np.asarray(occupied, dtype=bool)
        self.resolution = float(resolution)
        self.origin = (float(origin[0]), float(origin[1]))
        half = self.resolution / 2.0
        outside = distance_transform_edt(~occupied) * self.resolution - half
        inside = distance_transform_edt(occupied) * self.resolution - half
        signed = np.where(occupied, -inside, outside)
        self.distance = np.clip(signed, -max_distance, max_distance)
        gy, gx = np.gradient(self.distance, self.resolution)
        self.gradient_x, self.gradient_y = gx, gy

    def _coords(self, x, y):
        col = (np.asarray(x) - self.origin[0]) / self.resolution - 0.5
        row = (np.asarray(y) - self.origin[1]) / self.resolution - 0.5
        return np.vstack([row, col])

    def lookup(self, x, y):
        """Distance and gradient (dx, dy) at map points."""
        coords = self._coords(x, y)
        far = float(self.distance.max())
        d = map_coordinates(self.distance, coords, order=1, mode='constant', cval=far)
        gx = map_coordinates(self.gradient_x, coords, order=1, mode='constant', cval=0.0)
        gy = map_coordinates(self.gradient_y, coords, order=1, mode='constant', cval=0.0)
        return d, gx, gy


def occupancy_from_grid(data, width, height, threshold=65):
    """nav_msgs/OccupancyGrid data (row 0 = lowest y) -> boolean array."""
    grid = np.asarray(data, dtype=np.int16).reshape(height, width)
    return grid >= threshold


def compose(a, b):
    """Pose a then pose b (x, y, yaw)."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1],
            math.atan2(math.sin(a[2] + b[2]), math.cos(a[2] + b[2])))


def inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2])


def transform_points(pose, points):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    p = np.asarray(points, dtype=float)
    return np.column_stack([pose[0] + c * p[:, 0] - s * p[:, 1],
                            pose[1] + s * p[:, 0] + c * p[:, 1]])


def scan_points(ranges, angle_min, angle_increment, mount, odom_pose,
                range_min=0.1, range_max=30.0):
    """One LaserScan -> its points in the odom frame (n x 2).

    mount: scanner pose on the robot; odom_pose: robot pose in odom at the
    scan's own stamp.
    """
    r = np.asarray(ranges, dtype=float)
    a = angle_min + angle_increment * np.arange(len(r))
    ok = np.isfinite(r) & (r >= range_min) & (r <= range_max)
    local = np.column_stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])])
    return transform_points(compose(odom_pose, mount), local)


@dataclass
class Refinement:
    transform: tuple            # refined map -> odom (x, y, yaw)
    accepted: bool
    reason: str
    inliers: int
    inlier_fraction: float
    rms_m: float
    shift_m: float
    turn_rad: float
    constrained: int            # number of directions the scan pinned down (0-3)
    information: tuple          # eigenvalues of the information matrix


def _rotate_about(transform, centre, dyaw, dx, dy):
    """Turn a transform by dyaw about a map point, then shift it by (dx, dy)."""
    c, s = math.cos(dyaw), math.sin(dyaw)
    tx, ty = transform[0] - centre[0], transform[1] - centre[1]
    return (centre[0] + c * tx - s * ty + dx, centre[1] + s * tx + c * ty + dy,
            math.atan2(math.sin(transform[2] + dyaw), math.cos(transform[2] + dyaw)))


def refine(field, odom_points, prior, robot_odom, iterations=MAX_ITERATIONS):
    """Refine the map -> odom transform ``prior`` so odom_points lie on walls.

    The correction is solved about the robot (robot_odom: its position in the
    odom frame), not about the map origin: a heading correction then does not
    read as a shift, and "front-to-back" in the degeneracy test is the robot's.
    """
    pts = np.asarray(odom_points, dtype=float)
    x = tuple(float(v) for v in prior)
    eigvals = np.zeros(3)
    constrained = 0
    converged = False
    for _ in range(iterations):
        world = transform_points(x, pts)
        centre = transform_points(x, [robot_odom[:2]])[0]
        d, gx, gy = field.lookup(world[:, 0], world[:, 1])
        use = np.abs(d) < FAR_M
        if use.sum() < MIN_INLIERS:
            break
        r = d[use]
        w = np.where(np.abs(r) <= HUBER_M, 1.0, HUBER_M / np.maximum(np.abs(r), 1e-9))
        # Turning by dyaw about the robot moves a point by dyaw * (-(y-cy), x-cx).
        px, py = world[use, 0] - centre[0], world[use, 1] - centre[1]
        J = np.column_stack([gx[use], gy[use],
                             (-py * gx[use] + px * gy[use]) / ROTATION_LENGTH_M])
        H = (J * w[:, None]).T @ J
        g = (J * w[:, None]).T @ r
        eigvals, vecs = np.linalg.eigh(H)
        good = eigvals >= MIN_INFORMATION
        constrained = int(good.sum())
        if constrained == 0:
            break
        # Solve only in the constrained eigen-directions (solution remapping).
        coeff = np.zeros(3)
        coeff[good] = -(vecs[:, good].T @ g) / eigvals[good]
        step = vecs @ coeff
        dyaw = step[2] / ROTATION_LENGTH_M
        x = _rotate_about(x, centre, dyaw, step[0], step[1])
        if math.hypot(step[0], step[1]) < 1e-4 and abs(dyaw) < 1e-5:
            converged = True
            break
    world = transform_points(x, pts)
    d, _, _ = field.lookup(world[:, 0], world[:, 1])
    near = np.abs(d) < FAR_M
    inl = np.abs(d) < INLIER_M
    n_in = int(inl.sum())
    fraction = n_in / max(int(near.sum()), 1)
    rms = float(np.sqrt(np.mean(d[inl] ** 2))) if n_in else float('inf')
    # How far the ROBOT moves, not the transform's origin.
    before = compose(tuple(prior), tuple(robot_odom))
    after = compose(x, tuple(robot_odom))
    shift = math.hypot(after[0] - before[0], after[1] - before[1])
    turn = abs(math.atan2(math.sin(after[2] - before[2]), math.cos(after[2] - before[2])))
    x = list(x)
    checks = ((constrained == 0, 'no direction constrained'),
              (not converged, 'did not converge'),
              (n_in < MIN_INLIERS, f'{n_in} points on walls'),
              (fraction < MIN_INLIER_FRACTION, f'{fraction:.0%} of points on walls'),
              (rms > MAX_RMS_M, f'residual {rms * 100:.1f} cm'),
              (shift > MAX_SHIFT_M, f'{shift * 100:.1f} cm from AMCL'),
              (turn > MAX_TURN_RAD, f'{math.degrees(turn):.1f} deg from AMCL'))
    reason = next((text for failed, text in checks if failed), 'ok')
    x[2] = math.atan2(math.sin(x[2]), math.cos(x[2]))
    return Refinement(tuple(float(v) for v in x), reason == 'ok', reason, n_in, fraction,
                      rms, shift, turn, constrained, tuple(float(v) for v in eigvals))
