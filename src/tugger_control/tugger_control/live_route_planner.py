"""Live route planning with the offline generator's own algorithm and checks.

Plans from the robot's measured pose to a destination gate using the same
cardinal A*, Bezier smoothing, curvature limit and footprint validation that
produced the offline routes. The caller appends the gate's segment 2 approach.
"""

import math
import sys
import threading
from pathlib import Path


class LiveRoutePlanner:
    def __init__(self, tools_dir, map_yaml, *, world=None, clearance=1.05, corner_trim=1.0,
                 max_curvature=1.6, centre_weight=None):
        self.tools_dir = Path(tools_dir)
        self.map_yaml = Path(map_yaml)
        self.world = Path(world) if world is not None else None
        self.clearance = clearance
        self.corner_trim = corner_trim
        self.max_curvature = max_curvature
        self.centre_weight = centre_weight
        self._lock = threading.Lock()
        self._ready = None

    def _load(self):
        with self._lock:
            if self._ready is None:
                folder = str(self.tools_dir)
                if folder not in sys.path:
                    sys.path.insert(0, folder)
                import plan_routes as grid
                import build_route_database_v1 as generator
                from build_network import load_occupancy
                from scipy.ndimage import distance_transform_edt
                spec, safe = grid.load_map(self.map_yaml, self.clearance)
                occupancy_spec, blocked = load_occupancy(self.map_yaml)
                boxes = []
                if self.world is not None:
                    from tugger_control.warehouse_clearance import static_world_boxes
                    boxes = static_world_boxes(self.world)
                    keep_out = self._keep_out_grid(grid, spec, safe.shape, boxes)
                    # A live path starts anywhere, so it must avoid every modelled
                    # box, including stand legs the static map does not show.
                    resolution = float(spec['resolution'])
                    safe = safe & (distance_transform_edt(~keep_out) * resolution > self.clearance)
                    blocked = blocked | keep_out[:blocked.shape[0], :blocked.shape[1]]
                centre_cost = generator.centre_cost_map(blocked, float(spec['resolution']))
                self._ready = (grid, generator, spec, safe, occupancy_spec, blocked,
                               centre_cost, boxes)
            return self._ready

    @staticmethod
    def _keep_out_grid(grid, spec, shape, boxes):
        """Cells covered by world boxes, plus each stand's footprint from its legs."""
        import numpy as np
        mask = np.zeros(shape, dtype=bool)
        rects = [(b.x, b.y, b.half_x, b.half_y, b.yaw) for b in boxes]
        stands = {}
        for b in boxes:
            model = b.name.split('/')[0]
            if model.endswith('_legs'):
                stands.setdefault(model, []).append(b)
        for legs in stands.values():
            xs = [b.x for b in legs]
            ys = [b.y for b in legs]
            # The tabletop spans the legs; block that whole footprint.
            rects.append(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                          (max(xs) - min(xs)) / 2 + 0.1, (max(ys) - min(ys)) / 2 + 0.1, 0.0))
        resolution = float(spec['resolution'])
        for x, y, hx, hy, yaw in rects:
            reach = abs(hx * math.cos(yaw)) + abs(hy * math.sin(yaw)), \
                abs(hx * math.sin(yaw)) + abs(hy * math.cos(yaw))
            steps_x = int(reach[0] / resolution) + 1
            steps_y = int(reach[1] / resolution) + 1
            for i in range(-steps_x, steps_x + 1):
                for j in range(-steps_y, steps_y + 1):
                    try:
                        gx, gy = grid.cell([x + i * resolution, y + j * resolution], spec, shape)
                    except ValueError:
                        continue
                    mask[gy, gx] = True
        return mask

    def plan(self, start_pose, entries, keep_out=()):
        """Shortest valid route to any entry (gate staging pose).

        ``keep_out`` lists (x, y, radius) circles that must be avoided, for
        objects the map does not contain. Returns ``(poses, entry)``; raises
        ValueError if no entry is reachable.
        """
        grid, generator, spec, safe, occupancy_spec, blocked, centre_cost, boxes = self._load()
        if keep_out:
            safe = safe.copy()
            resolution = float(spec['resolution'])
            for x, y, radius in keep_out:
                reach = int(radius / resolution) + 1
                for i in range(-reach, reach + 1):
                    for j in range(-reach, reach + 1):
                        if i * i + j * j <= reach * reach:
                            try:
                                gx, gy = grid.cell([x + i * resolution, y + j * resolution],
                                                   spec, safe.shape)
                            except ValueError:
                                continue
                            safe[gy, gx] = False
        start = grid.cell([start_pose[0], start_pose[1]], spec, safe.shape)
        if not safe[start[1], start[0]]:
            raise ValueError('robot position is outside the clearance-safe planning map')
        best, last_error = None, 'no station entry configured'
        # Nearest entry first; the first valid route is used, to keep the
        # operator's wait short. Every route still passes all the checks.
        ordered = sorted(entries, key=lambda e: math.hypot(e[0] - start_pose[0],
                                                           e[1] - start_pose[1]))
        for entry in ordered:
            if best is not None:
                break
            try:
                goal = grid.cell([entry[0], entry[1]], spec, safe.shape)
                # Leave along the robot's body axis (forward or reverse): a
                # car-like 4WS robot cannot start a sideways first move.
                weight = (generator.CENTRE_WEIGHT if self.centre_weight is None
                          else self.centre_weight)
                candidate = generator.smooth_lead_out(
                    start, goal, safe, generator.direction_index(entry[2]),
                    generator.direction_index(start_pose[2]),
                    int(math.ceil(self.corner_trim / float(spec['resolution']))) + 1,
                    lambda cells: generator.smooth_and_validate(
                        cells, spec, safe, occupancy_spec, blocked,
                        self.corner_trim, self.max_curvature),
                    final_run=int(math.ceil(generator.FINAL_STRAIGHT_M /
                                            float(spec['resolution']))),
                    centre_cost=centre_cost, centre_weight=weight,
                    corner_penalty=generator.CORNER_PENALTY_M / float(spec['resolution']))
                self._audit(candidate['poses'], boxes)
                if best is None or candidate['length_m'] < best[0]['length_m']:
                    best = (candidate, entry)
            except Exception as exc:  # each entry is tried; report the last reason
                last_error = str(exc)
        if best is None:
            raise ValueError(f'live planning found no route: {last_error}')
        return self._finish(best)

    @staticmethod
    def _audit(poses, boxes):
        """Sampled 3D check of the whole path against every modelled box."""
        if not boxes:
            return
        from tugger_control.lift_geometry import LIFT_DOWN_M
        from tugger_control.precision_scene_clearance import sampled_precision_clearance
        clear, sample, hit = sampled_precision_clearance(
            [tuple(p) for p in poses[::5]] + [tuple(poses[-1])], boxes, LIFT_DOWN_M)
        if not clear:
            raise ValueError(f'path collides with {hit} near sample {sample}')

    @staticmethod
    def _finish(best):
        candidate, entry = best
        poses = [list(pose) for pose in candidate['poses']]
        if math.hypot(poses[-1][0] - entry[0], poses[-1][1] - entry[1]) > 0.03:
            raise ValueError('live route does not end on the station entry')
        return poses, entry
