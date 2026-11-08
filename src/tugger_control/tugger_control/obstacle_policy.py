"""What to do about an unexpected object: pure rules, no robot commands.

An object matters only if it would pass too close to the path still to be
driven. On a route the robot then waits, and goes around an object that stays;
at a station it pauses and alerts until the object is gone. The safety fields
are independent of this and can always stop the robot.
"""
import math

GAP_ROUTE_M = 0.5        # free space kept to an object on an open route
GAP_STATION_M = 0.15     # stations are tight: only a really close object counts
ROBOT_HALF_WIDTH_M = 0.33
CARGO_HALF_WIDTH_M = 0.68  # 1.3 m plate plus the 3 cm capture allowance
LOOK_AHEAD_M = 6.0
PREDICT_S = 3.0            # how far ahead a moving object's path is projected
PREDICT_STEP_S = 0.5
STATION_RADIUS_M = 3.0     # around the destination: an object there is a station matter
WAIT_BEFORE_AVOID_S = 10.0
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

CONTINUE, HOLD, ALERT, AVOID = 'CONTINUE', 'HOLD', 'ALERT', 'AVOID'


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


def interfering(obstacles, path, half_width, gap):
    """Objects whose edge is, or is about to be, within ``gap`` of the robot's edge."""
    return [o for o in obstacles
            if any(point_to_path(point, path) - o['r'] - half_width < gap
                   for point in predicted_positions(o))]


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
        if any(o.get('moving') for o in in_the_way):
            self.last_moving = now
            return AVOID if now - self.first_seen >= self.patience else HOLD
        # A standing object: the wait starts when it stopped moving.
        start = self.first_seen if self.last_moving is None else self.last_moving
        if not in_the_way or now - start < self.wait:
            return HOLD
        return AVOID


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
    nx, ny = -math.sin(tangent), math.cos(tangent)
    offset = (obstacle['x'] - points[nearest][0]) * nx + (obstacle['y'] - points[nearest][1]) * ny
    need = obstacle['r'] + half_width + gap + DETOUR_MARGIN_M
    moves = sorted((offset + need, offset - need), key=abs)
    centre = stations[nearest]
    for move in moves:
        ramp = max(MIN_RAMP_M, math.pi * math.sqrt(abs(move) / (2.0 * MAX_RAMP_CURVATURE)))
        hold = obstacle['r'] + 0.6          # robot length past the object either side
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
