#!/usr/bin/env python3
"""
AMR V1: Five-station, orientation-constrained route database.

Uses the current static warehouse map and existing Manhattan/
quintic-Bezier planner.

Features:
  - HOME plus four cargo stations
  - Four entrances evaluated per cargo station
  - Cardinal A* with a constrained terminal direction
  - Straight segments and quintic Bezier smoothing
  - Curvature, static clearance, and rectangular-footprint checks
  - Directed station-to-station route database
  - Saved individual paths
  - Explicit failures and validation metadata

The cargo staging positions are outside the stands.
This program DOES NOT plan entry beneath cargo, lift loads,
publish drive commands, or approve robot execution.
"""

import argparse
import hashlib
import heapq
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

import plan_routes as p

from build_network import (
    load_occupancy,
    validate_trajectory,
    footprint_is_free,
)


DIRECTIONS = ("east", "north", "west", "south")

# Heading is the direction of travel TOWARD the cargo.
APPROACH = {
    "east": math.pi,
    "north": -math.pi / 2,
    "west": 0.0,
    "south": math.pi / 2,
}

DIRS = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
)


def heading_error(a, b):
    return abs(math.atan2(
        math.sin(a - b),
        math.cos(a - b)
    ))


def path_length(poses):
    return sum(
        math.hypot(
            b[0] - a[0],
            b[1] - a[1]
        )
        for a, b in zip(poses, poses[1:])
    )


def direction_index(yaw):
    return min(
        range(4),
        key=lambda i: heading_error(
            math.atan2(DIRS[i][1], DIRS[i][0]),
            yaw
        )
    )


def centre_cost_map(blocked, resolution):
    """Per-cell cost of leaving the centre line of free space, 0 to 1.

    Routes are meant to keep equal distance to the obstacles on either side.
    The centre lines are the ridges of the distance-to-obstacle map: cells
    farther from obstacles than their neighbours across the corridor (a
    strict peak along x or y; the grid is axis-aligned, so this traces the
    generalised Voronoi diagram between racks, walls and stands). The cost is
    the Voronoi field share d_V / (d_O + d_V) (Dolgov et al., "Path Planning
    for Autonomous Vehicles in Unknown Semi-structured Environments", 2010):
    0 on a centre line, 1 at an obstacle, and relative to the local width, so
    the same offset costs more in a narrow gap than in an open crossing.
    """
    free = ~blocked
    d_obstacle = distance_transform_edt(free) * resolution
    p = np.pad(d_obstacle, 1)
    mid = p[1:-1, 1:-1]
    left, right, up, down = p[1:-1, :-2], p[1:-1, 2:], p[:-2, 1:-1], p[2:, 1:-1]
    along_x = (mid >= left) & (mid >= right) & ((mid > left) | (mid > right))
    along_y = (mid >= up) & (mid >= down) & ((mid > up) | (mid > down))
    ridge = free & (along_x | along_y) & (mid > 2.0 * resolution)
    if not ridge.any():
        return np.where(free, 0.0, 1.0)
    d_centre = distance_transform_edt(~ridge) * resolution
    cost = d_centre / np.maximum(d_obstacle + d_centre, 1e-9)
    cost[blocked] = 1.0
    return cost


def constrained_astar(start, goal, safe, terminal_direction,
                      centre_cost=None, centre_weight=0.0, initial_direction=-1,
                      corner_penalty=0.2):
    """
    Cardinal-grid A* with final direction as part of
    the goal condition.

    State: (grid_x, grid_y, incoming_direction).

    This constrains the terminal straight segment without
    adding an instantaneous rotation at the destination.
    """

    sx, sy = start
    gx, gy = goal

    h, w = safe.shape

    if not safe[sy, sx]:
        raise ValueError("Start cell not clearance-safe")

    if not safe[gy, gx]:
        raise ValueError("Goal cell not clearance-safe")

    initial = (sx, sy, initial_direction)

    best = {initial: 0.0}
    parent = {}

    queue = [
        (
            abs(gx - sx) + abs(gy - sy),
            0.0,
            initial
        )
    ]

    terminal = None

    while queue:
        _, cost, state = heapq.heappop(queue)

        if cost > best.get(state, float("inf")) + 1e-9:
            continue

        x, y, incoming = state

        if (
            (x, y) == (gx, gy)
            and incoming == terminal_direction
        ):
            terminal = state
            break

        for nd, (dx, dy) in enumerate(DIRS):
            xx = x + dx
            yy = y + dy

            if not (0 <= xx < w and 0 <= yy < h):
                continue

            if not safe[yy, xx]:
                continue

            # A direct 180-degree reversal is not modeled
            # as a smooth forward corner.
            if (
                incoming != -1
                and DIRS[incoming][0] == -dx
                and DIRS[incoming][1] == -dy
            ):
                continue

            turn_penalty = (
                corner_penalty
                if incoming != -1 and incoming != nd
                else 0.0
            )

            clearance_penalty = 0.0
            if centre_cost is not None and centre_weight > 0.0:
                # Prefer the centre line of free space (see centre_cost_map);
                # the hard footprint-safe boundary stays in ``safe``.
                clearance_penalty = centre_weight * float(centre_cost[yy, xx])
            new_cost = cost + 1.0 + turn_penalty + clearance_penalty
            nxt = (xx, yy, nd)

            if new_cost + 1e-9 < best.get(
                nxt, float("inf")
            ):
                best[nxt] = new_cost
                parent[nxt] = state

                heuristic = (
                    abs(gx - xx) + abs(gy - yy)
                )

                heapq.heappush(
                    queue,
                    (
                        new_cost + heuristic,
                        new_cost,
                        nxt
                    )
                )

    if terminal is None:
        raise ValueError(
            "No cardinal route reaches goal with "
            "the required terminal heading"
        )

    cells = []
    cursor = terminal

    while True:
        cells.append((cursor[0], cursor[1]))

        if cursor == initial:
            break

        cursor = parent[cursor]

    return cells[::-1]


def turn_limited_astar(start, goal, safe, terminal_direction, min_run, axis=None,
                       centre_cost=None, centre_weight=0.0, max_expansions=4000000,
                       corner_penalty=0.2,
                       inner_run=None, final_run=None):
    """Grid A* that only proposes corners the robot can drive.

    Every straight run between corners, the first from the start and the last
    into the goal, is at least ``min_run`` cells, so corner smoothing always
    has room for a curve within the turning limit. A plain shortest-cell
    search can return a one-cell jog or U-turn instead, which smoothing then
    rejects with no alternative. ``axis`` (a DIRS index) restricts the first
    move to the staging or body axis, either sign.

    Corner smoothing trims up to a corner length from each side, so a run
    between two corners needs ``inner_run`` (default twice ``min_run``);
    the first and last runs need only ``min_run``.

    State: (x, y, direction, straight cells so far (capped), first run?).
    """
    inner_run = 2 * min_run if inner_run is None else inner_run
    # The last run into the goal (a station gate) must let the path follower
    # settle its heading after the final corner.
    final_run = min_run if final_run is None else final_run
    cap = max(inner_run, final_run)
    sx, sy = start
    gx, gy = goal
    h, w = safe.shape
    if not safe[sy, sx] or not safe[gy, gx]:
        raise ValueError("Start or goal cell not clearance-safe")
    initial = (sx, sy, -1, 0, True)
    best = {initial: 0.0}
    parent = {}
    queue = [(abs(gx - sx) + abs(gy - sy), 0.0, initial)]
    expansions = 0
    while queue:
        _, cost, state = heapq.heappop(queue)
        if cost > best.get(state, float("inf")) + 1e-9:
            continue
        x, y, incoming, run, first = state
        if (x, y) == (gx, gy) and incoming == terminal_direction and run >= final_run:
            cells = []
            while True:
                cells.append((state[0], state[1]))
                if state == initial:
                    return cells[::-1]
                state = parent[state]
        expansions += 1
        if expansions > max_expansions:
            break
        for nd, (dx, dy) in enumerate(DIRS):
            if incoming == -1:
                if axis is not None and nd // 2 != axis // 2:
                    continue  # leave along the axis
            elif nd == incoming ^ 1:
                continue      # no reversal
            elif nd != incoming and run < (min_run if first else inner_run):
                continue      # no corner before a full straight run
            xx, yy = x + dx, y + dy
            if not (0 <= xx < w and 0 <= yy < h) or not safe[yy, xx]:
                continue
            new_run = min(run + 1, cap) if nd == incoming else 1
            new_first = first and (incoming == -1 or nd == incoming)
            penalty = corner_penalty if incoming != -1 and nd != incoming else 0.0
            if centre_cost is not None and centre_weight > 0.0:
                penalty += centre_weight * float(centre_cost[yy, xx])
            nxt = (xx, yy, nd, new_run, new_first)
            new_cost = cost + 1.0 + penalty
            if new_cost + 1e-9 < best.get(nxt, float("inf")):
                best[nxt] = new_cost
                parent[nxt] = state
                heapq.heappush(queue, (new_cost + abs(gx - xx) + abs(gy - yy), new_cost, nxt))
    raise ValueError("No turn-limited route reaches the goal")


def _coarse(grid, factor, reduce):
    """Downsample a boolean grid by ``factor`` (padding with False)."""
    h, w = grid.shape
    H, W = -(-h // factor), -(-w // factor)
    padded = np.zeros((H * factor, W * factor), dtype=bool)
    padded[:h, :w] = grid
    blocks = padded.reshape(H, factor, W, factor)
    return blocks.all(axis=(1, 3)) if reduce == "all" else blocks.any(axis=(1, 3))


def fast_turn_limited_astar(start, goal, safe, terminal_direction, min_run, axis=None,
                            centre_cost=None, centre_weight=0.0, factor=4, corner_penalty=0.2,
                            final_run=None):
    """Coarse-to-fine turn_limited_astar with the same exact result rules.

    A coarse grid (``factor`` cells per side) finds a corridor quickly; the
    exact fine-grid search then runs only inside that corridor. The coarse
    step keeps the centre-line preference (a block costs what its most
    central cell costs), so the corridor it hands on runs through the middle
    of free space like every other route. If even an optimistic coarse grid (a block
    is free if any of its cells is) has no turn-limited route, none exists
    and the costly full search is skipped.
    """
    final_run = min_run if final_run is None else final_run
    coarse_run = max(2, -(-min_run // factor))
    coarse_final = max(2, -(-final_run // factor))
    cs = (start[0] // factor, start[1] // factor)
    cg = (goal[0] // factor, goal[1] // factor)
    coarse_cost = None
    if centre_cost is not None and centre_weight > 0.0:
        h, w = centre_cost.shape
        H, W = -(-h // factor), -(-w // factor)
        padded = np.ones((H * factor, W * factor))
        padded[:h, :w] = centre_cost
        coarse_cost = padded.reshape(H, factor, W, factor).min(axis=(1, 3))
    corridor_path = None
    for reduce in ("all", "any"):
        coarse = _coarse(safe, factor, reduce)
        coarse[cs[1], cs[0]] = coarse[cg[1], cg[0]] = True
        try:
            corridor_path = turn_limited_astar(
                cs, cg, coarse, terminal_direction, coarse_run, axis=axis,
                centre_cost=coarse_cost,
                centre_weight=centre_weight * factor, final_run=coarse_final,
                corner_penalty=corner_penalty / factor)
            break
        except ValueError:
            continue
    if corridor_path is None:
        raise ValueError("No turn-limited route reaches the goal")
    h, w = safe.shape
    mask = np.zeros((-(-h // factor), -(-w // factor)), dtype=bool)
    for x, y in corridor_path:
        mask[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = True
    fine_mask = np.kron(mask, np.ones((factor, factor), dtype=bool))[:h, :w]
    for grid in (safe & fine_mask, safe):
        # The corridor first; everything only if it was too tight.
        try:
            return turn_limited_astar(start, goal, grid, terminal_direction, min_run,
                                      axis=axis, centre_cost=centre_cost,
                                      centre_weight=centre_weight,
                                      corner_penalty=corner_penalty,
                                      final_run=final_run)
        except ValueError:
            continue
    raise ValueError("No turn-limited route reaches the goal")


def start_is_sideways(cells, axis):
    """True if the first grid move is perpendicular to the staging axis."""
    if len(cells) < 2:
        return False
    move = (cells[1][0] - cells[0][0], cells[1][1] - cells[0][1])
    first = DIRS.index(move) if move in DIRS else axis
    # DIRS is east, west, north, south: pairs 0-1 and 2-3 share an axis.
    return first // 2 != axis // 2


def _runs(cells):
    runs = []
    for a, b in zip(cells, cells[1:]):
        move = (b[0] - a[0], b[1] - a[1])
        if runs and runs[-1][0] == move:
            runs[-1][1] += 1
        else:
            runs.append([move, 1])
    return runs


def _cells_from_runs(start, runs):
    cells = [tuple(start)]
    for (dx, dy), count in runs:
        for _ in range(count):
            cells.append((cells[-1][0] + dx, cells[-1][1] + dy))
    return cells


def _normalise(runs):
    """Merge equal neighbours and cancel opposite ones."""
    out = []
    for move, count in runs:
        if count <= 0:
            continue
        if out and out[-1][0] == move:
            out[-1][1] += count
        elif out and out[-1][0] == (-move[0], -move[1]):
            out[-1][1] -= count
            if out[-1][1] < 0:
                out[-1] = [move, -out[-1][1]]
            elif out[-1][1] == 0:
                out.pop()
        else:
            out.append([move, count])
    return out


def remove_short_jogs(cells, safe, max_jog):
    """Merge short sideways jogs into the next corner.

    A grid search drawn toward corridor centres can step one or two cells
    sideways between two runs in the same direction. Two corners that close
    cannot be smoothed into a drivable curve, so the sideways step is moved
    to the end of the following run (merging with the next corner) whenever
    every changed cell stays clearance-safe. Start and goal are unchanged.
    """
    runs = _runs(cells)
    h, w = safe.shape
    changed = True
    while changed:
        changed = False
        for i in range(1, len(runs) - 1):
            before, jog, after = runs[i - 1], runs[i], runs[i + 1]
            if before[0] != after[0] or jog[1] > max_jog:
                continue
            trial = _normalise(runs[:i - 1] + [[before[0], before[1] + after[1]], jog]
                               + runs[i + 2:])
            path = _cells_from_runs(cells[0], trial)
            if path[-1] == tuple(cells[-1]) and all(
                    0 <= x < w and 0 <= y < h and safe[y, x] for x, y in path):
                runs, changed = trial, True
                break
    return _cells_from_runs(cells[0], runs)


def lead_out_candidates(start, goal, safe, required, axis, run_cells, **weights):
    """Cell paths that leave the staging pose straight along its axis.

    Smoothing cannot round a corner at a path's first point, so a turn in
    the first cell stays a kink that a car-like 4WS robot cannot start from
    rest. Candidates run straight along the axis (either sign, since the
    robot may leave forward or in reverse; one and two corner lengths), then
    search on. Callers smooth each in turn and keep the first that passes.
    """
    plain = constrained_astar(start, goal, safe, required, **weights)
    first = _runs(plain)[:1]
    if first and DIRS.index(first[0][0]) // 2 == axis // 2 and first[0][1] >= run_cells:
        yield plain  # already leaves straight along the axis for a corner length
    options = []
    h, w = safe.shape
    # After the lead-out, the corridor-centre preference can pull the path
    # back across the lead (a one-cell U-turn); an unbiased search gives the
    # plain corner instead. Both are offered; smoothing decides.
    searches = [weights]
    if weights.get("centre_weight", 0.0) > 0.0:
        searches.append({**weights, "centre_weight": 0.0})
    for direction in (axis, axis ^ 1):
        dx, dy = DIRS[direction]
        for run in (run_cells, 2 * run_cells):
            lead = [(start[0] + k * dx, start[1] + k * dy) for k in range(run + 1)]
            if not all(0 <= x < w and 0 <= y < h and safe[y, x] for x, y in lead):
                continue
            for search in searches:
                try:
                    rest = constrained_astar(lead[-1], goal, safe, required,
                                             initial_direction=direction, **search)
                except ValueError:
                    continue
                options.append(lead + rest[1:])
    yield from sorted(options, key=len)


def search_with_lead_out(start, goal, safe, required, axis, run_cells, **weights):
    """The shortest lead-out candidate (see lead_out_candidates)."""
    for cells in lead_out_candidates(start, goal, safe, required, axis, run_cells, **weights):
        return cells
    raise ValueError("No route leaves the staging pose along its axis")


# Straight run into a gate after the final corner, so the path follower
# settles its heading before the docking handoff. Measured: 1.1 m arrived 12
# degrees off and stalled; about 1.6 m arrived reliably in every validation.
FINAL_STRAIGHT_M = 1.6
# A cell at an obstacle costs 1 + CENTRE_WEIGHT cell lengths, one on a centre
# line costs 1: a route takes a detour to the middle of free space when the
# detour is shorter than about twice the length it would otherwise run
# off-centre, so open crossings are crossed in the middle, not along a rack.
CENTRE_WEIGHT = 2.0
# Each corner slows the robot and swings its steering: worth about this much
# straight driving. Without it the search adds a side-step for any small bend
# of the centre line (measured: 1-2 m still let routes side-step 1 m in open
# space next to a rack end; 3 m does not, and crossings stay centred).
CORNER_PENALTY_M = 3.0


def smooth_lead_out(start, goal, safe, required, axis, run_cells, smooth, final_run=None,
                    **weights):
    """Smooth lead-out candidates in turn; return the first that validates."""
    last = "No route leaves the staging pose along its axis"

    def ordered():
        # 1. The plain shortest route, if it already leaves straight along
        #    the axis for a corner length (keeps existing good routes).
        try:
            plain = constrained_astar(start, goal, safe, required, **weights)
            runs = _runs(plain)
            if (runs and DIRS.index(runs[0][0]) // 2 == axis // 2 and
                    runs[0][1] >= run_cells and
                    (final_run is None or runs[-1][1] >= final_run)):
                yield plain
        except ValueError:
            pass
        # 2. The turn-limited search: only drivable corners, by construction.
        try:
            yield fast_turn_limited_astar(start, goal, safe, required, run_cells, axis=axis,
                                          final_run=final_run, **weights)
        except ValueError:
            pass
        # 3. Older lead-out candidates, built only if both fail.
        yield from lead_out_candidates(start, goal, safe, required, axis, run_cells, **weights)
    for cells in ordered():
        for attempt in (cells, None):
            if attempt is None:
                # Only when smoothing fails: merge short sideways jogs.
                attempt = remove_short_jogs(cells, safe, run_cells)
                if attempt == cells:
                    break
            try:
                return smooth(attempt)
            except ValueError as exc:
                last = str(exc)
    raise ValueError(last)


def align_route_ends(points, start_xy=None, goal_xy=None):
    """Move the first and last straight runs onto the exact staging poses.

    ``points`` are the compressed corner points of an axis-aligned route at
    grid cell centres. The end points become the exact poses and the corner
    point next to each end moves with them along the run's normal (at most
    half a cell), so every run stays axis-aligned and every corner stays 90
    degrees. A single run is only re-anchored when both poses lie on it.
    """
    out = [list(map(float, point)) for point in points]
    if len(out) < 2:
        return out

    def horizontal(a, b):
        return abs(b[1] - a[1]) <= abs(b[0] - a[0])

    if len(out) == 2:
        if start_xy is not None and goal_xy is not None:
            same_line = (abs(start_xy[1] - goal_xy[1]) < 1e-6 if horizontal(out[0], out[1])
                         else abs(start_xy[0] - goal_xy[0]) < 1e-6)
            if same_line:
                out[0][:2] = start_xy[:2]
                out[1][:2] = goal_xy[:2]
        return out
    if start_xy is not None:
        across = 1 if horizontal(out[0], out[1]) else 0
        out[0][:2] = start_xy[:2]
        out[1][across] = float(start_xy[across])
    if goal_xy is not None:
        across = 1 if horizontal(out[-2], out[-1]) else 0
        out[-1][:2] = goal_xy[:2]
        out[-2][across] = float(goal_xy[across])
    return out


def smooth_and_validate(
    cells,
    spec,
    safe,
    occupancy_spec,
    blocked,
    trim,
    curvature_limit,
    start_xy=None,
    goal_xy=None,
):
    points = align_route_ends(p.compress([
        p.world(q, spec)
        for q in cells
    ]), start_xy, goal_xy)

    if len(points) < 2:
        raise ValueError("Degenerate route")

    last_failure = "No valid smoothing"

    for factor in (1.0, 0.75, 0.5, 0.35):
        try:
            xy = p.smooth(
                points,
                trim * factor
            )

            k = p.curvature(xy)

            if k > curvature_limit:
                last_failure = (
                    f"Curvature {k:.3f} exceeds "
                    f"{curvature_limit:.3f}"
                )
                continue

            if not p.check_samples(xy, spec, safe):
                last_failure = (
                    "Smoothed route leaves clearance-safe map"
                )
                continue

            poses = []

            for i, point in enumerate(xy):
                if i + 1 < len(xy):
                    dx = xy[i + 1][0] - point[0]
                    dy = xy[i + 1][1] - point[1]
                else:
                    dx = point[0] - xy[i - 1][0]
                    dy = point[1] - xy[i - 1][1]

                poses.append([
                    round(point[0], 4),
                    round(point[1], 4),
                    round(math.atan2(dy, dx), 6)
                ])

            validation = validate_trajectory(
                poses,
                occupancy_spec,
                blocked
            )

            if not validation["valid"]:
                last_failure = (
                    "Rectangular footprint collision"
                )
                continue

            return {
                "poses": poses,
                "waypoints": points,
                "length_m": round(
                    path_length(poses), 4
                ),
                "max_curvature_1_per_m": round(k, 5),
                "trim_m": round(trim * factor, 4),
                "footprint_validation": validation,
            }

        except Exception as exc:
            last_failure = str(exc)

    raise ValueError(last_failure)


def staging_candidates(station, spec, safe, blocked, distance):
    """
    Construct staging poses outside each cargo stand.

    These are ordinary warehouse travel destinations,
    not positions underneath the cargo.
    """

    cx, cy = station["center"][:2]

    result = {}

    for direction in DIRECTIONS:
        yaw = APPROACH[direction]

        x = cx - distance * math.cos(yaw)
        y = cy - distance * math.sin(yaw)

        pose = [x, y, yaw]

        item = {
            "requested_pose": pose,
            "available": False,
            "reason": None,
        }

        try:
            grid = p.cell(
                [x, y],
                spec,
                safe.shape
            )

            if not safe[grid[1], grid[0]]:
                raise ValueError(
                    "Staging position fails static clearance"
                )

            # The staging pose stays exactly on the station axis. Snapping it
            # to the grid cell centre moved it 2.5 cm sideways, and the
            # station approach then bent back to the axis: a 2-3 cm S-bend at
            # both ends of every route (found 2026-11-12). The cell is only
            # the search start; align_route_ends() puts the route on the pose.
            if not footprint_is_free(
                pose,
                spec,
                blocked
            ):
                raise ValueError(
                    "Staging footprint blocked"
                )

            item.update({
                "available": True,
                "grid_cell": list(grid),
                "pose": pose,
                "reason": None,
            })

        except Exception as exc:
            item["reason"] = str(exc)

        result[direction] = item

    return result


def chain_successor(source, target, chains):
    """Return the stop visited after ``target`` on a ``source -> target`` leg."""
    for chain in chains:
        for before, current, after in zip(chain, chain[1:], chain[2:]):
            if before == source and current == target:
                return after
    return None


def chain_predecessor(source, target, chains):
    """Return the stop visited before ``source`` on a ``source -> target`` leg."""
    for chain in chains:
        for before, current, after in zip(chain, chain[1:], chain[2:]):
            if current == source and after == target:
                return before
    return None


def required_departure_entrance(source, target, known_routes, chains, fallback,
                                pass_through=()):
    """Choose the gate a route must leave by: the one its mission entered by.

    The environment has no way to enter by one gate and leave by another. A
    mission leg follows the route that delivered the robot; any other route
    falls back to the gate used when arriving from HOME.
    """
    if source in pass_through:
        return None
    before = chain_predecessor(source, target, chains)
    if before is not None:
        entrance = known_routes.get(
            f"{before}__TO__{source}", {}).get("target_entrance")
        if entrance is not None:
            return entrance
    return fallback.get(source)


def build_connection(
    source_name,
    target_name,
    nodes,
    spec,
    safe,
    occupancy_spec,
    blocked,
    args
):
    source_options = nodes[source_name]
    target_options = nodes[target_name]

    attempts = []
    winner = None
    valid = []

    for source_dir, source in source_options.items():
        if not source["available"]:
            continue
        # A gate variant leaves by a fixed gate; otherwise the mission
        # chain decides the gate this station is normally left by.
        preferred_source = getattr(args, "force_source_gate", None) or (
            required_departure_entrance(
                source_name,
                target_name,
                getattr(args, "known_routes", {}),
                getattr(args, "mission_chains", []),
                getattr(args, "departure_entrances", {}),
                getattr(args, "pass_through", ()),
            ))
        if preferred_source is not None and source_dir != preferred_source:
            continue

        for target_dir, target in target_options.items():
            if not target["available"]:
                continue

            attempt = {
                "source_entrance": source_dir,
                "target_entrance": target_dir,
                "status": "REJECTED",
            }

            try:
                start = tuple(source["grid_cell"])
                goal = tuple(target["grid_cell"])

                required = direction_index(
                    target["pose"][2]
                )

                candidate = smooth_lead_out(
                    start,
                    goal,
                    safe,
                    required,
                    direction_index(source["pose"][2]),
                    int(math.ceil(args.corner_trim / float(spec["resolution"]))) + 1,
                    lambda cells: smooth_and_validate(
                        cells, spec, safe, occupancy_spec, blocked,
                        args.corner_trim, args.max_curvature,
                        source["pose"][:2], target["pose"][:2]),
                    final_run=int(math.ceil(FINAL_STRAIGHT_M / float(spec["resolution"]))),
                    centre_cost=getattr(args, "centre_cost", None),
                    centre_weight=getattr(args, "centre_weight", 0.0),
                    corner_penalty=CORNER_PENALTY_M / float(spec["resolution"])
                )

                if heading_error(
                    candidate["poses"][-1][2],
                    target["pose"][2]
                ) > math.radians(2):
                    raise ValueError(
                        "Terminal heading mismatch"
                    )

                # Source heading is recorded as an
                # unresolved departure constraint unless
                # it happens to match this path.
                departure_error = heading_error(
                    candidate["poses"][0][2],
                    source["pose"][2]
                )
                # A route may leave a station in reverse.  Compare both the
                # forward and reverse body directions so an intentional
                # reverse departure is not rejected as a heading failure.
                departure_error = min(
                    departure_error,
                    abs(math.pi - departure_error)
                )

                attempt.update({
                    "status": "VALIDATED_STATIC_CANDIDATE",
                    "length_m": candidate["length_m"],
                    "departure_heading_error_deg": round(
                        math.degrees(departure_error), 2
                    ),
                })

                # Do not claim seamless station departure
                # when heading must change.
                candidate.update({
                    "source_entrance": source_dir,
                    "target_entrance": target_dir,
                    "source_pose": source["pose"],
                    "target_pose": target["pose"],
                    "departure_heading_error_deg": round(
                        math.degrees(departure_error), 2
                    ),
                    "departure_transition_validated": (
                        departure_error <= math.radians(2)
                    ),
                    "terminal_heading_validated": True,
                })

                valid.append(candidate)
                if (
                    winner is None
                    or candidate["length_m"] <
                    winner["length_m"]
                ):
                    winner = candidate

            except Exception as exc:
                # Clearance bias must not destroy an otherwise valid legacy
                # corridor. Retry the same entrance pair without the soft
                # preference; hard clearance and footprint checks remain.
                if getattr(args, "centre_weight", 0.0) > 0.0:
                    try:
                        candidate = smooth_lead_out(
                            start, goal, safe, required,
                            direction_index(source["pose"][2]),
                            int(math.ceil(args.corner_trim /
                                          float(spec["resolution"]))) + 1,
                            lambda cells: smooth_and_validate(
                                cells, spec, safe, occupancy_spec, blocked,
                                args.corner_trim, args.max_curvature,
                                source["pose"][:2], target["pose"][:2]),
                            final_run=int(math.ceil(FINAL_STRAIGHT_M /
                                                    float(spec["resolution"]))),
                            centre_cost=None, centre_weight=0.0)
                        departure_error = min(
                            heading_error(candidate["poses"][0][2], source["pose"][2]),
                            abs(math.pi - heading_error(
                                candidate["poses"][0][2], source["pose"][2])))
                        candidate.update({
                            "source_entrance": source_dir,
                            "target_entrance": target_dir,
                            "source_pose": source["pose"],
                            "target_pose": target["pose"],
                            "departure_heading_error_deg": round(
                                math.degrees(departure_error), 2),
                            "departure_transition_validated": (
                                departure_error <= math.radians(2)),
                            "terminal_heading_validated": True,
                        })
                        attempt.update({
                            "status": "VALIDATED_STATIC_CANDIDATE",
                            "length_m": candidate["length_m"],
                            "departure_heading_error_deg": round(
                                math.degrees(departure_error), 2),
                        })
                        valid.append(candidate)
                        if winner is None or candidate["length_m"] < winner["length_m"]:
                            winner = candidate
                        attempts.append(attempt)
                        continue
                    except Exception:
                        pass
                attempt["reason"] = str(exc)

            attempts.append(attempt)

    # In a mission chain the arrival gate decides how the next leg leaves
    # (same-gate rule), so choose it for this leg plus the next leg out of
    # it: a slightly shorter arrival must not force a much longer departure.
    after = chain_successor(source_name, target_name, getattr(args, "mission_chains", []))
    if (winner is not None and after is not None and
            not getattr(args, "_lookahead", False) and
            getattr(args, "force_source_gate", None) is None and
            target_name not in getattr(args, "pass_through", ())):
        onward = {}
        for gate in {candidate["target_entrance"] for candidate in valid}:
            look = argparse.Namespace(**vars(args))
            look._lookahead, look.force_source_gate = True, gate
            nxt = build_connection(target_name, after, nodes, spec, safe,
                                   occupancy_spec, blocked, look)
            onward[gate] = nxt["best"]["length_m"] if nxt.get("best") else float("inf")
        winner = min(valid, key=lambda c: c["length_m"] + onward[c["target_entrance"]])
        winner["chain_lookahead_m"] = round(onward[winner["target_entrance"]], 2)

    if winner is None:
        return {
            "status": "NO_VALIDATED_ROUTE",
            "attempts": attempts,
            "execution_approved": False,
        }

    return {
        "status": "VALIDATED_STATIC_CANDIDATE",
        "best": winner,
        "attempts": attempts,
        "execution_approved": False,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--map", required=True)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--output", required=True)

    ap.add_argument(
        "--home",
        nargs=2,
        type=float,
        default=[0.0, 0.0],
        help="HOME marker position; routes use gate staging poses around it"
    )

    ap.add_argument(
        "--home-gates",
        nargs="+",
        default=["east", "west"],
        choices=list(DIRECTIONS),
        help="HOME gates, aligned with the robot's HOME orientation"
    )

    ap.add_argument(
        "--clearance",
        type=float,
        default=1.05
    )

    ap.add_argument(
        "--staging-distance",
        type=float,
        default=2.0,
        help="Distance from cargo center to outer staging pose"
    )

    ap.add_argument(
        "--corner-trim",
        type=float,
        default=1.0
    )

    ap.add_argument(
        "--max-curvature",
        type=float,
        default=2.5
    )

    ap.add_argument(
        "--centre-weight",
        type=float,
        default=CENTRE_WEIGHT,
        help="A* cost per cell at an obstacle for leaving the centre line of "
             "free space (0 disables the preference)"
    )

    ap.add_argument(
        "--mission-chains",
        default=str(Path(__file__).with_name("mission_chains.json")),
        help="Mission order used to choose each route's departure gate"
    )

    ap.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="SOURCE__TO__TARGET",
        help="Regenerate only these routes and keep every other saved route"
    )

    ap.add_argument(
        "--gate-variants",
        action="store_true",
        help="Keep every saved route and add one route per other gate a station "
             "can be left by (a robot leaves by the gate it entered by)"
    )

    ap.add_argument(
        "--approaches",
        default=str(Path(__file__).resolve().parents[1] /
                    "mission_approaches/entrance_candidates.json"),
        help="Station approaches; a gate needs one to be entered and left by"
    )

    ap.add_argument(
        "--approve-regenerated",
        action="store_true",
        help="Mark routes regenerated by --only as approved for execution"
    )

    args = ap.parse_args()

    if min(
        args.clearance,
        args.staging_distance,
        args.corner_trim,
        args.max_curvature
    ) <= 0:
        ap.error("Planning distances/limits must be positive")
    if args.centre_weight < 0:
        ap.error("Centre weight must be non-negative")

    started = time.monotonic()

    geometry = json.loads(
        Path(args.geometry).read_text()
    )

    spec, safe = p.load_map(
        Path(args.map),
        args.clearance
    )

    occupancy_spec, blocked = load_occupancy(
        args.map
    )
    args.centre_cost = centre_cost_map(blocked, float(occupancy_spec["resolution"]))

    home_cell = p.cell(
        args.home,
        spec,
        safe.shape
    )

    home_xy = p.world(home_cell, spec)

    if not safe[home_cell[1], home_cell[0]]:
        raise ValueError("HOME is not clearance-safe")

    if not footprint_is_free(
        [home_xy[0], home_xy[1], 0.0],
        occupancy_spec,
        blocked
    ):
        raise ValueError("HOME footprint blocked")

    # HOME is reached through gate staging poses like a station. Its own
    # segment 2 then ends exactly on the marker, which is off the grid.
    home_gates = staging_candidates(
        {"center": list(args.home)}, spec, safe, blocked, args.staging_distance)
    nodes = {
        "HOME": {gate: home_gates[gate] for gate in args.home_gates}
    }

    for name, station in geometry["stations"].items():
        nodes[name] = staging_candidates(
            station,
            spec,
            safe,
            blocked,
            args.staging_distance
        )

    names = list(nodes)

    # Preserve the validated entrance used to arrive at each cargo station.
    # This makes a station-to-station route begin at the same physical
    # departure side without hard-coding station IDs into the planner.
    args.departure_entrances = {}
    args.known_routes = {}
    chain_config = json.loads(Path(args.mission_chains).read_text())
    args.mission_chains = chain_config["chains"]
    args.pass_through = tuple(chain_config.get("pass_through", ()))
    previous = None
    previous_database = Path(args.output) / "route_database.json"
    if previous_database.exists():
        try:
            previous = json.loads(previous_database.read_text())
            args.known_routes = dict(previous.get("routes", {}))
            for station in names:
                entry = previous.get("routes", {}).get(
                    f"HOME__TO__{station}", {})
                entrance = entry.get("target_entrance")
                if entrance in nodes.get(station, {}):
                    args.departure_entrances[station] = entrance
        except (OSError, ValueError, TypeError):
            args.departure_entrances = {}

    if len(names) != 5:
        raise ValueError(
            f"Expected HOME + four cargo stations, "
            f"found {len(names)}"
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    import yaml

    map_yaml = Path(args.map)
    map_info = yaml.safe_load(map_yaml.read_text())

    image_file = map_yaml.parent / map_info["image"]

    map_hash = hashlib.sha256(
        map_yaml.read_bytes() +
        image_file.read_bytes()
    ).hexdigest()

    database = {
        "version": "AMR_V1_FIVE_STATION_DATABASE",
        "map_sha256": map_hash,
        "stations": names,
        "staging_distance_m": args.staging_distance,
        "clearance_radius_m": args.clearance,
        "centre_weight": args.centre_weight,
        "nodes": nodes,
        # Stops without a stand; the robot may leave them by any gate.
        "pass_through": list(args.pass_through),
        # Exact centres, so segment 2 tools need no hard-coded positions.
        "station_centers": {
            "HOME": [float(args.home[0]), float(args.home[1])],
            **{name: [float(v) for v in station["center"][:2]]
               for name, station in geometry["stations"].items()},
        },
        "routes": {},
        "execution_approved": False,
        "notes": [
            "Routes connect warehouse staging poses",
            "Cargo approach and lift are not included",
            "Terminal headings checked",
            "Departure transitions require validation",
            "No dynamic obstacles modeled",
            "Robot control and localization not validated"
        ]
    }

    total = len(names) * (len(names) - 1)
    number = 0
    accepted = 0

    print()
    print("========== AMR V1 ROUTE DATABASE ==========")
    print("Warehouse map:", args.map)
    print("Stations:", len(names))
    print("Directed connections:", total)
    print("All four cargo entrances considered")
    print("Robot movement: DISABLED")
    print("===========================================")

    print("\nSTAGING POSITION CHECK")

    for name, options in nodes.items():
        valid = [
            direction
            for direction, data in options.items()
            if data["available"]
        ]

        print(f"  {name}: {valid}")

    print("\nGENERATING ROUTES")

    if args.gate_variants:
        if previous is None:
            raise ValueError("--gate-variants needs an existing route_database.json")
        for name in ("routes", "execution_approved", "summary"):
            database[name] = previous[name]
        approaches = json.loads(Path(args.approaches).read_text())["candidates"]
        gates = {}
        for key in approaches:
            station, gate = key.rsplit("__", 1)
            gates.setdefault(station, []).append(gate)
        args.variants = []
        for source in names:
            if source in args.pass_through:
                continue  # left by any gate already
            for gate in sorted(gates.get(source, [])):
                for target in names:
                    primary = previous["routes"].get(f"{source}__TO__{target}", {})
                    if target == source or primary.get("source_entrance") == gate:
                        continue
                    args.variants.append(
                        (source, target, f"{source}__{gate}__TO__{target}", gate))
        if args.only:
            # Replan only the named gate variants; every other route is kept.
            args.variants = [v for v in args.variants if v[2] in args.only]
            args.only = []
        total = len(args.variants)

    if args.only:
        if previous is None:
            raise ValueError("--only needs an existing route_database.json")
        known = {f"{a}__TO__{b}" for a in names for b in names if a != b}
        unknown = set(args.only) - known
        if unknown:
            raise ValueError(f"Unknown routes for --only: {sorted(unknown)}")
        for name in ("routes", "execution_approved", "summary"):
            database[name] = previous[name]
        total = len(args.only)

    # Mission legs first, in order, so each sees the gate its predecessor chose.
    chained = [pair for chain in args.mission_chains
               for pair in zip(chain, chain[1:])]
    ordered = chained + [(a, b) for a in names for b in names if a != b]
    planned = [(a, b, f"{a}__TO__{b}", None) for a, b in dict.fromkeys(ordered)]
    if args.gate_variants:
        planned = args.variants

    # Gate variants do not depend on each other: build them on every core.
    precomputed = {}
    if args.gate_variants and len(planned) > 1:
        import os
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
            jobs = {}
            for source, target, key, forced_gate in planned:
                job_args = argparse.Namespace(**vars(args))
                job_args.force_source_gate = forced_gate
                jobs[key] = pool.submit(build_connection, source, target, nodes, spec,
                                        safe, occupancy_spec, blocked, job_args)
            precomputed = {key: job.result() for key, job in jobs.items()}

    for source, target, key, forced_gate in planned:
        if True:
            if args.only and key not in args.only:
                continue
            args.force_source_gate = forced_gate

            number += 1

            print(
                f"\n[{number}/{total}] "
                f"{source} -> {target}",
                flush=True
            )

            result = precomputed.get(key) or build_connection(
                source,
                target,
                nodes,
                spec,
                safe,
                occupancy_spec,
                blocked,
                args
            )

            entry = {
                "source": source,
                "target": target,
                "status": result["status"],
                "attempts": result["attempts"],
                "execution_approved": False
            }
            if forced_gate is not None:
                entry["gate_variant_of"] = f"{source}__TO__{target}"
            approve_new = bool(args.only and args.approve_regenerated)

            if result["status"] == (
                "VALIDATED_STATIC_CANDIDATE"
            ):
                best = result["best"]
                accepted += 1

                path_file = output / f"{key}.json"

                path_file.write_text(
                    json.dumps(
                        {
                            "version": "AMR_V1_SAVED_ROUTE",
                            "map_sha256": map_hash,
                            "source": source,
                            "target": target,
                            **best,
                            "execution_approved": approve_new
                        },
                        indent=2
                    ) + "\n"
                )

                entry.update({
                    "execution_approved": approve_new,
                    "route_file": path_file.name,
                    "length_m": best["length_m"],
                    "source_entrance": best["source_entrance"],
                    "target_entrance": best["target_entrance"],
                    "terminal_heading_validated": True,
                    "departure_transition_validated": (
                        best["departure_transition_validated"]
                    )
                })

                print(
                    "  PASS:",
                    round(best["length_m"], 2),
                    "m | terminal:",
                    best["target_entrance"],
                    "| departure continuous:",
                    best["departure_transition_validated"],
                    flush=True
                )

            else:
                print(
                    "  NO VALIDATED ROUTE",
                    flush=True
                )

            database["routes"][key] = entry
            args.known_routes[key] = entry

    if args.only or args.gate_variants:
        accepted = sum(
            route.get("status") == "VALIDATED_STATIC_CANDIDATE"
            for route in database["routes"].values())
        total = len(database["routes"])

    database["summary"] = {
        "connections_requested": total,
        "static_candidates_found": accepted,
        "connections_without_candidate": total - accepted,
        "elapsed_seconds": round(
            time.monotonic() - started, 2
        ),
        "complete_mission_execution_approved": False
    }

    database_file = output / "route_database.json"

    temporary = database_file.with_suffix(
        ".json.tmp"
    )

    temporary.write_text(
        json.dumps(database, indent=2) + "\n"
    )

    temporary.replace(database_file)

    print("\n========== DATABASE SUMMARY ==========")
    print("Requested connections:", total)
    print("Validated static candidates:", accepted)
    print("Unresolved connections:", total - accepted)
    print("Elapsed:", database["summary"]["elapsed_seconds"], "s")
    print("Saved:", database_file)
    print("Execution approved:",
          "SELECTED ROUTES ONLY" if args.only and args.approve_regenerated else "NO")
    print("=======================================")


if __name__ == "__main__":
    main()
