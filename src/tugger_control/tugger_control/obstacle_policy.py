"""What to do about an unexpected object: pure rules, no robot commands.

An object matters only if it would pass too close to the path still to be
driven. On a route the robot then waits, and goes around an object that stays;
at a station it pauses and alerts until the object is gone. The safety fields
are independent of this and can always stop the robot.
"""
import math

GAP_ROUTE_M = 0.5        # free space kept to an object on an open route at full speed
# A standing object closer than GAP_ROUTE_M but at least GAP_ROUTE_MIN_M from
# the robot's (or its cargo's) side is passed at SLOW_PASS_SPEED_M_S (user
# decision 2026-11-14); closer than that it is in the way. A moving object
# (a person) keeps the full GAP_ROUTE_M. The speed is an assumption, no
# standard value: slow enough that the safety fields stay small.
GAP_ROUTE_MIN_M = 0.15
SLOW_PASS_SPEED_M_S = 0.3
GAP_STATION_M = 0.15     # stations are tight: only a really close object counts
ROBOT_HALF_WIDTH_M = 0.33
CARGO_HALF_WIDTH_M = 0.68  # 1.3 m plate plus the 3 cm capture allowance
LOOK_AHEAD_M = 6.0
PREDICT_S = 3.0            # how far ahead a moving object's path is projected
PREDICT_STEP_S = 0.5
STATION_RADIUS_M = 3.0     # around the destination: an object there is a station matter
WAIT_BEFORE_AVOID_S = 5.0
MOVING_PATIENCE_S = 30.0   # a moving object blocking this long: look for another path
CLEAR_BEFORE_RESUME_S = 3.0
# While paused an object must clear by this much more, so a reading near a limit
# cannot flip the robot between waiting and driving.
HOLD_EXTRA_GAP_M = 0.10
HOLD_EXTRA_LOOK_AHEAD_M = 2.0
MAX_RAMP_CURVATURE = 0.8   # 1/m: gentle against the 4WS turning limit
MIN_RAMP_M = 1.5
DETOUR_OBJECT_RADIUS_M = 4.5   # objects this close to the one avoided must also be cleared
DETOUR_WORLD_MARGIN_M = 0.30   # from walls and racks: outside the safety field
DETOUR_MARGIN_M = 0.15     # extra room: a measured object's size jitters by centimetres

# Objects the colour cameras name as people or forklifts (obstacle_recognition)
# may start moving at any moment: even standing still they are treated as
# moving - full gap, waited for (patience) instead of driven around after the
# short wait (user decision 2026-11-08: people and forklifts are waited for).
MAY_MOVE_CLASSES = ('person', 'forklift')
# Look before deciding (user decision 2026-11-20): the scanners can see an
# object around a corner long before the colour cameras can. An unnamed
# standing object the cameras have not yet had in view LOOKS_BEFORE_JUDGING
# times is not judged (no 5 s clock, no detour) while it is further along the
# path than LOOK_STOP_M from the robot's centre to the object's near side; the
# robot drives on at SLOW_PASS_SPEED_M_S so the cameras can look. Closer, it is
# handled as today. With no recognition running, objects carry no 'looks' and
# are handled as today.
LOOKS_BEFORE_JUDGING = 2
LOOK_STOP_M = 1.5          # 0.47 m to the front bumper: about 1 m of floor left
# Back up, then go around (user decision 2026-11-20): when a detour cannot start
# from where the robot stands (too close to bend out), it reverses back along
# the floor it just drove, at BACKUP_SPEED_M_S and at most BACKUP_MAX_M, to the
# nearest point from which the detour fits. Once per object; if still blocked,
# the route network is replanned.
BACKUP_MAX_M = 2.0
BACKUP_STEP_M = 0.25
BACKUP_SPEED_M_S = 0.3
BACKUP_STRAIGHT_TOLERANCE_M = 0.05   # a driven stretch this close to a line is backed straight
# Backed up just far enough, the detour no longer fitted: the route goal checker
# accepts a stop 15 cm short (nav2.yaml goal_checker) and localization jitters
# by a few cm, so the robot backs up this much further (campaign 6, 2026-11-20:
# 0.75 m planned, 0.5-0.65 m driven).
BACKUP_MARGIN_M = 0.25

CONTINUE, HOLD, ALERT, AVOID = 'CONTINUE', 'HOLD', 'ALERT', 'AVOID'


def may_move(obstacle):
    """Moving now, or a person or forklift that could move at any moment."""
    return bool(obstacle.get('moving')) or obstacle.get('cls') in MAY_MOVE_CLASSES


def awaiting_look(obstacle):
    """Unnamed, standing, and not yet judged by the colour cameras: looked at
    fewer than LOOKS_BEFORE_JUDGING times, or seen as something once and
    being recognised (half cycle 2026-11-20: a person seen 3.2 s apart while
    the robot turned was judged unnamed and briefly passed like a box)."""
    looks = obstacle.get('looks')
    return (looks is not None and not obstacle.get('cls') and not obstacle.get('moving')
            and (looks < LOOKS_BEFORE_JUDGING or bool(obstacle.get('pending'))))


def unidentified(obstacle):
    """Recognition is running but has not named it: it could be a person, so
    it keeps the full gap and is never passed closely (user, 2026-11-20).
    Without recognition (no 'looks') objects are handled as before."""
    return obstacle.get('looks') is not None and not obstacle.get('cls')


def distance_along(path, point):
    """Distance along ``path`` (from its start) to the path point nearest ``point``."""
    if not path:
        return 0.0
    nearest = min(range(len(path)), key=lambda i: math.dist(path[i][:2], point))
    return sum(math.dist(a[:2], b[:2]) for a, b in zip(path[:nearest], path[1:nearest + 1]))


def split_for_look(in_way, ahead, stop=LOOK_STOP_M):
    """(still in the way, to approach slowly so the cameras can look): an object
    awaiting a look is approached while its near side is more than ``stop``
    along the path ``ahead`` (which starts at the robot)."""
    judged, approach = [], []
    for o in in_way:
        far = distance_along(ahead, (o['x'], o['y'])) - o.get('r', 0.0) > stop
        (approach if awaiting_look(o) and far else judged).append(o)
    return judged, approach


def points_behind(path, robot_xy, distance):
    """The driven path from the robot back ``distance`` metres, nearest first,
    or None if the path behind is shorter."""
    if not path:
        return None
    start = min(range(len(path)), key=lambda i: math.dist(path[i][:2], robot_xy))
    out, travelled = [tuple(robot_xy)], 0.0
    previous = tuple(robot_xy)
    for i in range(start, -1, -1):
        point = tuple(path[i][:2])
        step = math.dist(previous, point)
        if travelled + step >= distance:
            if step > 0:
                f = (distance - travelled) / step
                out.append((previous[0] + f * (point[0] - previous[0]),
                            previous[1] + f * (point[1] - previous[1])))
            return out
        if step > 0:
            out.append(point)
            travelled += step
            previous = point
    return None


def backup_line(behind, step=0.05, tolerance=BACKUP_STRAIGHT_TOLERANCE_M):
    """Straight back when the driven stretch is straight, else along it (user,
    2026-11-20): points from the robot to the back-up point every ``step``."""
    (ax, ay), (bx, by) = behind[0], behind[-1]
    length = math.hypot(bx - ax, by - ay)
    straight = length > 0 and all(
        abs((px - ax) * (by - ay) - (py - ay) * (bx - ax)) / length <= tolerance
        for px, py in behind)
    if not straight:
        return list(behind)
    n = max(1, int(math.ceil(length / step)))
    return [(ax + (bx - ax) * k / n, ay + (by - ay) * k / n) for k in range(n + 1)]


def plan_backup(path, robot_xy, obstacle, half_width, is_clear,
                max_distance=BACKUP_MAX_M, step=BACKUP_STEP_M, margin=BACKUP_MARGIN_M):
    """The shortest back-up along the driven path after which a detour fits:
    (back-up points, detour points), or None if none within ``max_distance``."""
    distance = step
    while distance <= max_distance + 1e-9:
        behind = points_behind(path, robot_xy, distance)
        if behind is None:
            return None
        if plan_detour(path, behind[-1], obstacle, half_width, is_clear) is not None:
            # Further back by the stopping margin (within the limit and the
            # driven path), so a stop a little short still leaves room.
            for extra in (margin, margin / 2, 0.0):
                total = min(distance + extra, max_distance)
                behind = points_behind(path, robot_xy, total)
                if behind is None:
                    continue
                bent = plan_detour(path, behind[-1], obstacle, half_width, is_clear)
                if bent is not None:
                    return backup_line(behind), bent
        distance += step
    return None


def path_length(points):
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def path_ahead(path, robot_xy, length=LOOK_AHEAD_M):
    """The part of ``path`` from the point nearest the robot, up to ``length``."""
    if not path:
        return []
    start = min(range(len(path)), key=lambda i: math.dist(path[i][:2], robot_xy))
    ahead, travelled = [path[start]], 0.0
    for a, b in zip(path[start:], path[start + 1:]):
        travelled += math.dist(a[:2], b[:2])
        ahead.append(b)
        if travelled >= length:
            break
    return ahead


def point_to_path(point, path):
    """Shortest distance from ``point`` to the polyline ``path``."""
    if len(path) == 1:
        return math.dist(point, path[0][:2])
    best = float('inf')
    for a, b in zip(path, path[1:]):
        ax, ay, bx, by = a[0], a[1], b[0], b[1]
        dx, dy = bx - ax, by - ay
        span = dx * dx + dy * dy
        t = 0.0 if span == 0 else max(0.0, min(1.0, ((point[0] - ax) * dx + (point[1] - ay) * dy) / span))
        best = min(best, math.hypot(point[0] - ax - t * dx, point[1] - ay - t * dy))
    return best


def predicted_positions(obstacle):
    """Where an object is now and, if it moves, over the next few seconds."""
    points = [(obstacle['x'], obstacle['y'])]
    if obstacle.get('moving'):
        vx, vy = obstacle.get('vx', 0.0), obstacle.get('vy', 0.0)
        steps = int(PREDICT_S / PREDICT_STEP_S)
        points += [(obstacle['x'] + vx * k * PREDICT_STEP_S, obstacle['y'] + vy * k * PREDICT_STEP_S)
                   for k in range(1, steps + 1)]
    return points


def edge_distance(obstacle, path, centre=None):
    """Distance from the object's nearest edge to ``path``.

    With an outline (the shape the sensors saw) that is the nearest outline
    point; a circle of half the longest extent would reach past the sides of
    a long object. ``centre`` moves the object there (a predicted position).
    """
    cx, cy = centre if centre is not None else (obstacle['x'], obstacle['y'])
    shape = obstacle.get('outline')
    if not shape:
        return point_to_path((cx, cy), path) - obstacle['r']
    dx, dy = cx - obstacle['x'], cy - obstacle['y']
    return (min(point_to_path((px + dx, py + dy), path) for px, py in shape) -
            obstacle.get('outline_margin', 0.03))


def interfering(obstacles, path, half_width, gap):
    """Objects whose edge is, or is about to be, within ``gap`` of the robot's edge."""
    return [o for o in obstacles
            if any(edge_distance(o, path, point) - half_width < gap
                   for point in predicted_positions(o))]


def route_objects(obstacles, path, half_width, extra_gap=0.0):
    """(in the way, to pass slowly) on a route: standing objects need only
    GAP_ROUTE_MIN_M, moving ones, people and forklifts, and objects the cameras
    have not named yet the full GAP_ROUTE_M."""
    full_gap = [o for o in obstacles if may_move(o) or unidentified(o)]
    standing = [o for o in obstacles if not (may_move(o) or unidentified(o))]
    in_way = (interfering(full_gap, path, half_width, GAP_ROUTE_M + extra_gap) +
              interfering(standing, path, half_width, GAP_ROUTE_MIN_M + extra_gap))
    blocked = {id(o) for o in in_way}
    slow = [o for o in interfering(standing, path, half_width, GAP_ROUTE_M)
            if id(o) not in blocked]
    return in_way, slow


def effective_phase(phase, in_the_way, destination_centre):
    """An object at the station being driven to is handled as a station object."""
    if phase == 'route' and destination_centre is not None and any(
            math.hypot(o['x'] - destination_centre[0], o['y'] - destination_centre[1]) <= STATION_RADIUS_M
            for o in in_the_way):
        return 'station'
    return phase


class ObstacleSupervisor:
    """Turn what is in the way into an action, with the waiting times."""

    def __init__(self, wait=WAIT_BEFORE_AVOID_S, clear=CLEAR_BEFORE_RESUME_S,
                 patience=MOVING_PATIENCE_S):
        self.wait, self.clear, self.patience = wait, clear, patience
        self.first_seen, self.last_seen, self.last_moving = None, None, None

    def reset(self):
        self.first_seen = self.last_seen = self.last_moving = None

    def update(self, now, phase, in_the_way):
        """``phase`` is 'route' or 'station'; returns the action to take."""
        if in_the_way:
            if self.first_seen is None:
                self.first_seen = now
            self.last_seen = now
        elif self.last_seen is None or now - self.last_seen >= self.clear:
            self.reset()
            return CONTINUE
        if phase == 'station':
            return ALERT
        if any(may_move(o) for o in in_the_way):
            self.last_moving = now
            return AVOID if now - self.first_seen >= self.patience else HOLD
        # A standing object: the wait starts when it stopped moving.
        start = self.first_seen if self.last_moving is None else self.last_moving
        if not in_the_way or now - start < self.wait:
            return HOLD
        return AVOID


def object_extent(obstacle, origin, tangent):
    """How far an object reaches to the left and right of the route line through
    ``origin`` (along ``tangent``), and half its length along it: (left, right,
    half length). From its measured outline when it has one: the enclosing circle
    reaches past a long object's sides (a 1.2 x 0.8 m pallet along the aisle was
    taken as 1.40 m wide and no detour fitted, campaign 5, 2026-11-20)."""
    tx, ty = math.cos(tangent), math.sin(tangent)
    nx, ny = -ty, tx
    shape = obstacle.get('outline')
    if shape:
        margin = obstacle.get('outline_margin', 0.03)
        across = [(px - origin[0]) * nx + (py - origin[1]) * ny for px, py in shape]
        along = [(px - origin[0]) * tx + (py - origin[1]) * ty for px, py in shape]
        mid = (obstacle['x'] - origin[0]) * tx + (obstacle['y'] - origin[1]) * ty
        return (max(across) + margin, min(across) - margin,
                max(abs(a - mid) for a in along) + margin)
    offset = (obstacle['x'] - origin[0]) * nx + (obstacle['y'] - origin[1]) * ny
    return offset + obstacle['r'], offset - obstacle['r'], obstacle['r']


def plan_detour(path, robot_xy, obstacle, half_width, is_clear, gap=GAP_ROUTE_M):
    """Bend the path sideways around one object and back onto the route.

    Tries the side needing the smaller move first. Smooth cosine ramps keep the
    curvature within what car-like steering follows. Returns (x, y) points from
    the robot to the end of ``path``, or None if neither side is clear.
    """
    start = min(range(len(path)), key=lambda i: math.dist(path[i][:2], robot_xy))
    points = [tuple(p[:2]) for p in path[start:]]
    if len(points) < 2:
        return None
    stations = [0.0]
    for a, b in zip(points, points[1:]):
        stations.append(stations[-1] + math.dist(a, b))
    nearest = min(range(len(points)), key=lambda i: math.dist(points[i], (obstacle['x'], obstacle['y'])))
    a = points[max(nearest - 1, 0)]
    b = points[min(nearest + 1, len(points) - 1)]
    tangent = math.atan2(b[1] - a[1], b[0] - a[0])
    base = half_width + gap + DETOUR_MARGIN_M
    left, right, length = object_extent(obstacle, points[nearest], tangent)
    moves = sorted((left + base, right - base), key=abs)
    centre = stations[nearest]
    for move in moves:
        ramp = max(MIN_RAMP_M, math.pi * math.sqrt(abs(move) / (2.0 * MAX_RAMP_CURVATURE)))
        hold = length + 0.6                 # robot length past the object either side
        begin, end = centre - hold - ramp, centre + hold + ramp
        if begin < 0.3 or end > stations[-1] - 0.3:
            continue
        bent = []
        for point, s, i in zip(points, stations, range(len(points))):
            if s <= begin or s >= end:
                bent.append(point)
                continue
            if s < begin + ramp:
                w = 0.5 * (1 - math.cos(math.pi * (s - begin) / ramp))
            elif s > end - ramp:
                w = 0.5 * (1 - math.cos(math.pi * (end - s) / ramp))
            else:
                w = 1.0
            p = points[max(i - 1, 0)]
            q = points[min(i + 1, len(points) - 1)]
            t = math.atan2(q[1] - p[1], q[0] - p[0])
            bent.append((point[0] - math.sin(t) * move * w, point[1] + math.cos(t) * move * w))
        if is_clear(bent):
            return bent
    return None
