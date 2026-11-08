"""Bounded, motion-free 4WS visual-alignment primitive search.

This planner does not publish commands and is not a collision safety layer.
Its caller must supply a fresh, conservative clearance predicate and must
independently guard, monitor, and stop any executed motion.
"""

import math
import heapq
import itertools
import time

# Fraction of the position tolerance that a preferred plan must end within.
TIGHT_FRACTION = 0.5


def propagate(pose, distance, curvature):
    """Exact counter-phase 4WS arc: yaw rate = velocity * curvature."""
    x, y, yaw = pose
    change = distance * curvature
    if abs(curvature) < 1e-9:
        return x + distance * math.cos(yaw), y + distance * math.sin(yaw), yaw
    return (x + (math.sin(yaw + change) - math.sin(yaw)) / curvature,
            y - (math.cos(yaw + change) - math.cos(yaw)) / curvature,
            math.atan2(math.sin(yaw + change), math.cos(yaw + change)))


def advance(pose, primitive, distance=None):
    """Pose after ``distance`` (default: all) of an arc or crab primitive."""
    length = primitive[0] if distance is None else distance
    if len(primitive) > 2 and primitive[2] != 0.0:
        from tugger_control.crab_motion import crab_propagate
        return crab_propagate(pose, length, primitive[2])
    return propagate(pose, length, primitive[1])


def crab_macros(target):
    """Crab moves (distance, 0, wheel angle) that remove the lateral error.

    One diagonal move straight at the target when its direction is within
    the wheel limit; otherwise the steepest diagonal forward or backward,
    whose leftover forward error the caller's straight correction removes,
    or a forward-and-back pair that ends with no forward change.
    """
    from tugger_control.crab_motion import CRAB_MAX_ANGLE_RAD
    tx, ty = target[0], target[1]
    if abs(ty) < 0.003:
        return []
    macros = []
    distance = math.hypot(tx, ty)
    for sign in (1.0, -1.0):
        beta = math.atan2(sign * ty, sign * tx)
        if abs(beta) <= CRAB_MAX_ANGLE_RAD and distance >= 0.005:
            macros.append(((sign * distance, 0.0, beta),))
    steep = CRAB_MAX_ANGLE_RAD
    side = math.copysign(1.0, ty)
    reach = abs(ty) / math.sin(steep)
    for sign in (1.0, -1.0):
        # Forward-diagonal (sign +1) or backward-diagonal (sign -1).
        macros.append(((sign * reach, 0.0, sign * side * steep),))
        half = reach / 2.0
        if half >= 0.005:
            macros.append(((sign * half, 0.0, sign * side * steep),
                           (-sign * half, 0.0, -sign * side * steep)))
    return [m for m in macros if all(0.005 <= abs(p[0]) <= 0.10 for p in m)]


def replan_visual_maneuver(pose, target, clearance, **options):
    """Replan from a measured pose; targets and clearance are in its world frame.

    This is motion-free and does not validate observation freshness, scene
    completeness, or heading calibration. The caller must do so separately.
    """
    pose = tuple(float(value) for value in pose)
    target = tuple(float(value) for value in target)
    if (len(pose) != 3 or len(target) != 3 or
            not all(math.isfinite(value) for value in (*pose, *target)) or
            not callable(clearance)):
        raise ValueError('finite poses and a clearance predicate are required')
    c, s = math.cos(pose[2]), math.sin(pose[2])
    dx, dy = target[0] - pose[0], target[1] - pose[1]
    local_target = (c * dx + s * dy, -s * dx + c * dy,
                    math.atan2(math.sin(target[2] - pose[2]),
                               math.cos(target[2] - pose[2])))

    def world_clearance(local_pose):
        x, y, yaw = local_pose
        return clearance((pose[0] + c * x - s * y,
                          pose[1] + s * x + c * y,
                          math.atan2(math.sin(pose[2] + yaw),
                                     math.cos(pose[2] + yaw))))

    return plan_visual_maneuver(local_target, world_clearance, **options)


def marker_visible(pose, target, *, height=0.06, horizontal_fov=2.44346,
                   aspect=0.75, marker_size=0.12, margin=0.004):
    """Conservative pinhole test of all square corners in the camera image.

    Assumes a level ground plane and camera centred over the robot. Target
    yaw is expressed in the same frame as pose; unknown yaw must not be guessed.
    """
    x, y, yaw = pose
    tx, ty, tyaw = target
    half = marker_size / 2
    horizontal = height * math.tan(horizontal_fov / 2) - margin
    vertical = height * math.tan(horizontal_fov / 2) * aspect - margin
    for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        wx = tx + half * (sx * math.cos(tyaw) - sy * math.sin(tyaw))
        wy = ty + half * (sx * math.sin(tyaw) + sy * math.cos(tyaw))
        dx, dy = wx - x, wy - y
        forward = math.cos(yaw) * dx + math.sin(yaw) * dy
        lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
        if abs(forward) > vertical or abs(lateral) > horizontal:
            return False
    return True


def _strict_visible_plan(target, clearance, *, max_depth=6, beam_width=220,
                         position_tolerance=0.018, heading_tolerance=0.10):
    """Search finite forward/reverse, straight/steered arcs from base origin.

    `clearance(pose)` must check the robot and any attached cargo against a
    validated scene. Each candidate is sampled every 5 mm. Returns a list of
    (signed metres, curvature) or None. Completeness is not guaranteed.
    """
    target = tuple(float(value) for value in target)
    if len(target) != 3 or not all(math.isfinite(value) for value in target):
        raise ValueError('finite target x, y and heading required')
    if not callable(clearance):
        raise ValueError('clearance predicate required')
    if not clearance((0., 0., 0.)) or not marker_visible((0., 0., 0.), target):
        return None
    primitives = [(direction * length, curvature)
                  for length in (0.025, 0.05)
                  for direction in (-1, 1)
                  for curvature in (-3.0, 0.0, 3.0)]

    def cost(pose):
        return (math.hypot(target[0] - pose[0], target[1] - pose[1]) +
                0.04 * abs(math.atan2(math.sin(target[2] - pose[2]),
                                      math.cos(target[2] - pose[2]))))

    frontier = [((0., 0., 0.), ())]
    for _ in range(max_depth):
        candidates = []
        for pose, path in frontier:
            for distance, curvature in primitives:
                # Reject immediate exact reversal of the same arc.
                if path and path[-1] == (-distance, curvature):
                    continue
                count = math.ceil(abs(distance) / 0.005)
                samples = [propagate(pose, distance * index / count, curvature)
                           for index in range(1, count + 1)]
                if not all(marker_visible(sample, target) and clearance(sample)
                           for sample in samples):
                    continue
                end = samples[-1]
                candidate = path + ((distance, curvature),)
                heading = abs(math.atan2(math.sin(target[2] - end[2]),
                                         math.cos(target[2] - end[2])))
                if math.hypot(target[0] - end[0], target[1] - end[1]) <= position_tolerance and heading <= heading_tolerance:
                    return candidate
                candidates.append((cost(end), end, candidate))
        candidates.sort(key=lambda entry: entry[0])
        frontier = [(pose, path) for _, pose, path in candidates[:beam_width]]
        if not frontier:
            break
    return None


def plan_visual_maneuver(target, clearance, *, max_depth=6, beam_width=220,
                         position_tolerance=0.018,
                         heading_tolerance=0.10,
                         max_expansions=1800, time_budget=3.0,
                         require_full_visibility=True):
    """Bounded best-first search of signed 4WS arcs, sampled at <=5 mm.

    By default retain the strict camera-visible search for legacy tests.
    With require_full_visibility=False, apply the expanded local maneuver
    search. Each *executed* candidate is swept against exact clearance. The
    marker is checked at the start and final target; it need not remain fully
    in frame at *every hypothetical search sample*. At runtime, missing or
    stale camera feedback MUST stop motion and force reacquisition. This
    function never authorizes blind travel and has no completeness guarantee.
    """
    if require_full_visibility:
        # Preserve the previous strict planner for existing offline validation
        # and callers that depend on all corners staying in view at all times.
        return _strict_visible_plan(target, clearance, max_depth=max_depth,
                                    beam_width=beam_width,
                                    position_tolerance=position_tolerance,
                                    heading_tolerance=heading_tolerance)
    target = tuple(float(v) for v in target)
    if (len(target) != 3 or not all(math.isfinite(v) for v in target) or
            not callable(clearance)):
        raise ValueError('finite target and clearance predicate required')
    if not clearance((0.0, 0.0, 0.0)) or not marker_visible((0.0, 0.0, 0.0), target):
        return None
    if max_depth < 1 or beam_width < 1 or max_expansions < 1 or time_budget <= 0:
        raise ValueError('invalid search budgets')
    # Four-arc parallel-parking macros are particularly important for a 4WS
    # robot targeting a nearby lateral displacement. A naïve greedy search
    # keeps selecting low-error states and never discovers the necessary
    # temporary detour. Each macro finishes with its starting yaw.
    def swept(origin, actions):
        pose = origin
        for primitive in actions:
            distance = primitive[0]
            count = math.ceil(abs(distance) / .005)
            for index in range(1, count + 1):
                sample = advance(pose, primitive, distance * index / count)
                # Require complete red-frame visibility throughout each arc.
                if not marker_visible(sample, target, height=.056) or not clearance(sample):
                    return None
            pose = advance(pose, primitive)
        return pose

    # Explore short 4WS parallel-parking maneuvers plus one measured yaw
    # correction. Smaller arcs are critical after steering-slew leaves a few
    # millimetres of residual error at the end of the first sequence.
    sign = 1 if target[1] >= 0 else -1
    macros = []
    for length in (.025, .035, .038, .039, .04, .045, .05):
        macros.extend((
            # Alternate directions to limit longitudinal excursion.
            ((-length, -3*sign), (length, 3*sign),
             (length, -3*sign), (-length, 3*sign)),
            ((length, -3*sign), (-length, 3*sign),
             (-length, -3*sign), (length, 3*sign)),
            ((length, 3*sign), (length, -3*sign),
             (-length, 3*sign), (-length, -3*sign)),
            ((length, -3*sign), (-length, 3*sign),
             (-length, -3*sign), (length, 3*sign)),
            ((-length, -3*sign), (-length, 3*sign),
             (length, -3*sign), (length, 3*sign)),
        ))
    # Crab first: one diagonal move (all wheels at one angle) per macro.
    crabs = crab_macros(target)
    macros = crabs + macros
    candidates = []
    # Correct longitudinal displacement before lateral parking if useful.
    prefixes = [()]
    if .005 <= abs(target[0]) <= .05:
        prefixes.append(((target[0], 0.0),))
    for prefix in prefixes:
        first = swept((0.0, 0.0, 0.0), prefix)
        if first is None:
            continue
        for macro in ((), *macros):
            origin = first
            steps = prefix
            for repetitions in range(
                    1 + 1 if macro in crabs else
                    (max_depth - len(prefix)) // 4 + 1 if macro else 1):
                if repetitions:
                    origin = swept(origin, macro)
                    if origin is None:
                        break
                    steps += macro
                if len(steps) > max_depth:
                    break
                yaw_delta = math.atan2(math.sin(target[2] - origin[2]),
                                       math.cos(target[2] - origin[2]))
                # Use variable-curvature short arcs for exact yaw correction.
                # Direction of the final arc is selected with the forward error.
                trims = [()]
                if 1e-6 < abs(yaw_delta) <= .15:
                    trims = []
                    for direction in (-1, 1):
                        amount = max(.005, abs(yaw_delta) / 3.0)
                        if amount <= .05:
                            distance = direction * amount
                            trims.append(((distance, yaw_delta / distance),))
                for trim in trims:
                    if len(steps) + len(trim) > max_depth:
                        continue
                    end = swept(origin, trim) if trim else origin
                    if end is None:
                        continue
                    dx = target[0] - end[0]
                    dy = target[1] - end[1]
                    forward_delta = math.cos(end[2])*dx + math.sin(end[2])*dy
                    if abs(forward_delta) < .005:
                        corrections = ()
                    elif abs(forward_delta) <= .05:
                        corrections = ((forward_delta, 0.0),)
                    elif abs(forward_delta) <= .1:
                        corrections = ((math.copysign(.05, forward_delta), 0.0),
                                       (forward_delta - math.copysign(.05, forward_delta), 0.0))
                    else:
                        continue
                    if len(steps) + len(trim) + len(corrections) > max_depth:
                        continue
                    final = swept(end, corrections) if corrections else end
                    if final is None or not marker_visible(final, target, height=.056):
                        continue
                    d = math.hypot(final[0] - target[0], final[1] - target[1])
                    h = abs(math.atan2(math.sin(final[2] - target[2]),
                                       math.cos(final[2] - target[2])))
                    if d <= position_tolerance and h <= heading_tolerance:
                        proposal = steps + trim + corrections
                        if proposal:
                            # Each change of wheel pattern costs a stop and
                            # a re-steer: count it like 2 cm of travel.
                            switches = sum(
                                1 for a, b in zip(proposal, proposal[1:])
                                if (len(a) > 2 and a[2] != 0.0) != (len(b) > 2 and b[2] != 0.0)
                                or (len(a) > 2 and len(b) > 2 and a[2] != b[2]))
                            candidates.append((sum(abs(t[0]) for t in proposal) +
                                               0.02 * switches,
                                               len(proposal), d, proposal))
    if candidates:
        # Prefer plans that end well inside the tolerance. Picking the
        # shortest plan alone chose manoeuvres that end right at the
        # tolerance edge, so millimetres of tracking and camera error left the
        # robot just outside it after every correction ("no progress").
        tight = [c for c in candidates if c[2] <= position_tolerance * TIGHT_FRACTION]
        if tight:
            return min(tight)[3]
        return min(candidates, key=lambda c: (c[2], c[0], c[1]))[3]
    def error(pose):
        d = math.hypot(target[0] - pose[0], target[1] - pose[1])
        a = abs(math.atan2(math.sin(target[2] - pose[2]),
                           math.cos(target[2] - pose[2])))
        return d, a

    def key(pose):
        return (round(pose[0] / 0.005), round(pose[1] / 0.005),
                round(pose[2] / 0.02))

    # 5 mm terminal steps make an actual 10 mm / 1 degree target attainable;
    # longer arcs handle the bulk translation with fewer segment stops.
    primitives = [(direction * length, curvature)
                  for length in (0.025, 0.05, 0.01, 0.005)
                  for direction in (-1, 1)
                  for curvature in (-3.0, 0.0, 3.0)]
    # Best-first, heuristic-dominated search, with capped depth and expansions.
    # Beam width bounds queue pressure, not just final depth candidates.
    counter = itertools.count()
    start = (0.0, 0.0, 0.0)
    frontier = [(0.0, next(counter), start, (), 0.0)]
    seen = {key(start): 0.0}
    deadline = time.monotonic() + time_budget
    expansions = 0
    fallback = None   # best plan inside the tolerance but not the tight band
    while frontier and expansions < max_expansions and time.monotonic() < deadline:
        _, _, pose, path, traveled = heapq.heappop(frontier)
        if len(path) >= max_depth:
            continue
        expansions += 1
        for distance, curvature in primitives:
            if time.monotonic() >= deadline:
                break
            if path and path[-1] == (-distance, curvature):
                continue
            end = propagate(pose, distance, curvature)
            d, a = error(end)
            # Keep the local capture search confined. A Nav2 reposition is
            # needed outside this envelope; visual servo does not drive blind.
            if abs(end[0]) > .16 or abs(end[1]) > .14 or abs(end[2]) > .55:
                continue
            # Complete marker must be measurable at the goal, not necessarily
            # at intermediate *predicted* samples. Runtime enforces freshness.
            if d <= position_tolerance and a <= heading_tolerance and not marker_visible(end, target, height=.056):
                continue
            next_traveled = traveled + abs(distance)
            cell = key(end)
            if seen.get(cell, float('inf')) <= next_traveled - 0.002:
                continue
            steps = math.ceil(abs(distance) / 0.005)
            if not all(
                    marker_visible(propagate(pose, distance * i / steps, curvature),
                                   target, height=.056)
                    and clearance(propagate(pose, distance * i / steps, curvature))
                    for i in range(1, steps + 1)):
                continue
            candidate = path + ((distance, curvature),)
            if d <= position_tolerance and a <= heading_tolerance:
                if d <= position_tolerance * TIGHT_FRACTION:
                    return candidate
                if fallback is None or d < fallback[0]:
                    fallback = (d, candidate)
            seen[cell] = next_traveled
            # Weight distance and heading errors heavily to reach a solution
            # before exhausting the bounded search. Subtle length penalty
            # discourages unnecessarily long or cyclic camera maneuvers.
            heuristic = d + .12 * a
            priority = heuristic + .08 * next_traveled + .001 * len(candidate)
            heapq.heappush(frontier, (priority, next(counter), end, candidate,
                                      next_traveled))
        if len(frontier) > beam_width * 3:
            frontier = heapq.nsmallest(beam_width, frontier)
            heapq.heapify(frontier)
    return fallback[1] if fallback is not None else None
