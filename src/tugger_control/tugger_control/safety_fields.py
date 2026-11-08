"""Speed-dependent safety fields for the tugger (ISO 3691-4 style).

Modelled on how certified AGV safety controllers use a safety laser scanner
(SICK nanoScan3 class) and 3D cameras:

* Protective field: covers the distance the robot needs to stop from its
  current speed (sensor response + controller reaction + braking) plus the
  scanner's measurement supplement and a tolerance. An object inside it stops
  the robot (protective stop); it restarts automatically once the field has
  been clear for RESTART_CLEAR_S.
* Warning field: a longer field ahead of it; an object inside slows the robot
  to WARNING_SPEED.
* Near stations (docking, departure, camera alignment, turning in place) the
  robot moves at most STATION_SPEED, and reduced fields apply, as ISO 3691-4
  permits at load transfer stations.
* While turning in place the field is a ring around the swept circle.

Fields follow the arc the robot is driving (its steering curvature), so a
curve does not see the racks beside the straight line ahead. Values marked
ASSUMED are engineering choices, not taken from the standard's text.
Pure geometry; no robot commands.
"""

import math

# Robot body (chassis 0.88 x 0.58 m; wheels inside it).
HALF_LENGTH = 0.44
HALF_WIDTH = 0.29

# Sensor and controller timing.
SCANNER_RESPONSE_S = 0.070     # nanoScan3 datasheet: 70 ms
CONTROL_REACTION_S = 0.100     # ASSUMED: gate 20 Hz + controller 20 Hz
SCANNER_SUPPLEMENT_M = 0.065   # nanoScan3 protective field supplement
FIELD_TOLERANCE_M = 0.10       # ASSUMED: localisation/measurement tolerance
# Braking used to size fields: lower than the controller's 1.5 m/s^2 so a
# worn or wet floor still stops inside the field (ASSUMED).
SAFETY_DECELERATION = 1.0

# Speeds.
MAX_SPEED = 1.0                # user decision, mixed traffic
WARNING_SPEED = 0.3            # ASSUMED: creep while someone is nearby
STATION_SPEED = 0.3            # ISO 3691-4 practice at transfer stations
RESTART_CLEAR_S = 2.0          # ASSUMED (Dematic quotes about 3 s)

SIDE_MARGIN = 0.10             # protective field beyond the body sides
WARNING_SIDE_MARGIN = 0.30
WARNING_EXTRA_S = 1.5          # warning field = protective + 1.5 s of travel
WARNING_MIN_EXTRA_M = 0.5
STATION_SIDE_MARGIN = 0.03     # reduced field at stations (legs pass beside)

# States, in increasing severity.
CLEAR, WARNING, PROTECTIVE_STOP, EMERGENCY_STOP = (
    'CLEAR', 'WARNING', 'PROTECTIVE_STOP', 'EMERGENCY_STOP')


def stopping_distance(speed, deceleration=SAFETY_DECELERATION):
    """Travel from detection to standstill at ``speed``."""
    speed = abs(speed)
    return speed * (SCANNER_RESPONSE_S + CONTROL_REACTION_S) + speed * speed / (
        2.0 * deceleration)


def protective_length(speed):
    """Protective field length beyond the body in the direction of travel."""
    return stopping_distance(speed) + SCANNER_SUPPLEMENT_M + FIELD_TOLERANCE_M


def warning_length(speed):
    return protective_length(speed) + max(WARNING_MIN_EXTRA_M, abs(speed) * WARNING_EXTRA_S)


def _in_arc_band(x, y, direction, curvature, length, half_width):
    """Is body-frame point (x, y) inside the band swept ahead along an arc?

    The band starts at the body edge in the direction of travel (sign of
    ``direction``) and runs ``length`` along a path of ``curvature``
    (1/m, positive turning left), ``half_width`` either side of it.
    """
    if direction < 0:
        x, y, curvature = -x, y, -curvature   # mirror: reversing
    if x < -HALF_LENGTH:
        return False
    # Distance along the path ahead of the front edge, and offset from it.
    if abs(curvature) < 1e-6:
        along, offset = x - HALF_LENGTH, y
    else:
        radius = 1.0 / curvature
        # Path from the front edge (HALF_LENGTH, 0), centre at (HALF_LENGTH, radius).
        dx, dy = x - HALF_LENGTH, y - radius
        angle = math.atan2(dx, -dy if radius > 0 else dy)
        along = angle * abs(radius)
        offset = abs(radius) - math.hypot(dx, dy)
        if radius < 0:
            offset = -offset
    if along > length:
        return False
    if along < 0:
        # Beside the body: the side margin only (the body itself is excluded).
        return abs(y) <= half_width and x >= -HALF_LENGTH
    return abs(offset) <= half_width


def inside_body(x, y, margin=0.02):
    return abs(x) <= HALF_LENGTH + margin and abs(y) <= HALF_WIDTH + margin


def _arc_band_mask(x, y, direction, curvature, length, half_width):
    """Vectorised _in_arc_band for numpy arrays x, y."""
    import numpy as np
    if direction < 0:
        x, curvature = -x, -curvature
    inside = x >= -HALF_LENGTH
    if abs(curvature) < 1e-6:
        along, offset = x - HALF_LENGTH, y
    else:
        radius = 1.0 / curvature
        dx, dy = x - HALF_LENGTH, y - radius
        angle = np.arctan2(dx, -dy if radius > 0 else dy)
        along = angle * abs(radius)
        offset = abs(radius) - np.hypot(dx, dy)
    ahead = (along >= 0) & (along <= length) & (np.abs(offset) <= half_width)
    beside = (along < 0) & (np.abs(y) <= half_width)
    return inside & (ahead | beside)


def classify_points(points, speed, curvature, *, station=False, turning=False,
                    turn_radius=0.0):
    """Return (state, a point in the most severe field, or None).

    ``points`` are obstacle points (x, y) in the robot body frame, already
    filtered for height. Points on the robot's own body are ignored.
    """
    import numpy as np
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return CLEAR, None
    x, y = pts[:, 0], pts[:, 1]
    off_body = ~((np.abs(x) <= HALF_LENGTH + 0.02) & (np.abs(y) <= HALF_WIDTH + 0.02))
    if turning:
        r = np.hypot(x, y)
        protect = turn_radius + SCANNER_SUPPLEMENT_M
        hits = off_body & (r <= protect)
        if hits.any():
            i = int(np.argmax(hits))
            return PROTECTIVE_STOP, (float(x[i]), float(y[i]))
        warn = off_body & (r <= protect + 0.3)
        if warn.any():
            i = int(np.argmax(warn))
            return WARNING, (float(x[i]), float(y[i]))
        return CLEAR, None
    direction = 1.0 if speed >= 0 else -1.0
    if station:
        speed = min(abs(speed), STATION_SPEED)
        side = warn_side = HALF_WIDTH + STATION_SIDE_MARGIN
        protect_len = warn_len = protective_length(speed)
    else:
        side, warn_side = HALF_WIDTH + SIDE_MARGIN, HALF_WIDTH + WARNING_SIDE_MARGIN
        # Fields never shrink below those for creeping, so a stopped robot
        # still guards the space it is about to move into.
        protect_len = protective_length(max(abs(speed), WARNING_SPEED))
        warn_len = warning_length(max(abs(speed), WARNING_SPEED))
    hits = off_body & _arc_band_mask(x, y, direction, curvature, protect_len, side)
    if hits.any():
        i = int(np.argmax(hits))
        return PROTECTIVE_STOP, (float(x[i]), float(y[i]))
    warn = off_body & _arc_band_mask(x, y, direction, curvature, warn_len, warn_side)
    if warn.any():
        i = int(np.argmax(warn))
        return WARNING, (float(x[i]), float(y[i]))
    return CLEAR, None


def classify_crab(points, speed):
    """Fields for a crab move (all wheels at one angle, any direction).

    The robot translates diagonally at low speed, so the protective field is
    its footprint grown by the protective length on every side, and the
    warning field 0.3 m beyond that. Returns like classify_points.
    """
    import numpy as np
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return CLEAR, None
    x, y = pts[:, 0], pts[:, 1]
    off_body = ~((np.abs(x) <= HALF_LENGTH + 0.02) & (np.abs(y) <= HALF_WIDTH + 0.02))
    grow = protective_length(abs(speed))
    for state, extra in ((PROTECTIVE_STOP, 0.0), (WARNING, 0.3)):
        hits = off_body & (np.abs(x) <= HALF_LENGTH + grow + extra) & (
            np.abs(y) <= HALF_WIDTH + grow + extra)
        if hits.any():
            i = int(np.argmax(hits))
            return state, (float(x[i]), float(y[i]))
    return CLEAR, None


def speed_limit(state, station=False):
    """Largest speed the safety state allows."""
    if state in (PROTECTIVE_STOP, EMERGENCY_STOP):
        return 0.0
    if station or state == WARNING:
        return min(MAX_SPEED, STATION_SPEED if station else WARNING_SPEED)
    return MAX_SPEED


class RestartTimer:
    """Protective stops restart automatically once clear for RESTART_CLEAR_S;
    emergency stops (e-stop button, bumper) only by an explicit reset."""

    def __init__(self, clear_s=RESTART_CLEAR_S):
        self.clear_s = clear_s
        self.stopped_since_clear = None
        self.held = False
        self.emergency = None

    def update(self, measured, now):
        """Return the effective state for the latest measured field state."""
        if self.emergency is not None:
            return EMERGENCY_STOP
        if measured == PROTECTIVE_STOP:
            self.held, self.stopped_since_clear = True, None
            return PROTECTIVE_STOP
        if self.held:
            if self.stopped_since_clear is None:
                self.stopped_since_clear = now
            if now - self.stopped_since_clear < self.clear_s:
                return PROTECTIVE_STOP
            self.held = False
        return measured

    def trigger_emergency(self, reason):
        if self.emergency is None:
            self.emergency = reason

    def reset(self):
        """Operator reset; refused while a cause is still active (caller checks)."""
        self.emergency = None
        self.held, self.stopped_since_clear = True, None
