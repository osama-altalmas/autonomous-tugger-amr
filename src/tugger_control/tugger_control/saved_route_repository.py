"""Read-only access to the offline-generated tugger route database."""

import json
import hashlib
import math
from pathlib import Path


class SavedRouteRepository:
    """Load only static-validated routes produced before simulation."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.database = json.loads(
            (self.directory / 'route_database.json').read_text(encoding='utf-8'))
        if self.database.get('version') != 'AMR_V1_FIVE_STATION_DATABASE':
            raise ValueError('Unsupported saved-route database')

    def route(self, key):
        entry = self.database.get('routes', {}).get(key)
        if not entry:
            raise ValueError(f'Unknown saved route: {key}')
        if entry.get('status') != 'VALIDATED_STATIC_CANDIDATE':
            raise ValueError(f'Saved route is not statically validated: {key}')
        filename = entry.get('route_file')
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f'Invalid saved route filename: {key}')
        route = json.loads((self.directory / filename).read_text(encoding='utf-8'))
        if route.get('source') != entry.get('source') or route.get('target') != entry.get('target'):
            raise ValueError(f'Saved route identity mismatch: {key}')
        # Preserve the database gate separately: a route file cannot override it.
        route['_database_execution_approved'] = entry.get('execution_approved') is True
        poses = route.get('poses')
        if not isinstance(poses, list) or len(poses) < 2:
            raise ValueError(f'Saved route has insufficient poses: {key}')
        converted = [[float(value) for value in pose] for pose in poses]
        if any(len(pose) != 3 or not all(math.isfinite(value) for value in pose)
               for pose in converted):
            raise ValueError(f'Saved route has invalid pose data: {key}')
        return route, converted

    def routes_to(self, destination):
        """Return every statically-valid saved route ending at ``destination``."""
        matches = []
        for key, entry in self.database.get('routes', {}).items():
            if entry.get('target') == destination:
                route, poses = self.route(key)
                matches.append((key, route, poses))
        if not matches:
            raise ValueError(f'No saved routes target destination: {destination}')
        return matches


class PrecisionDockingRepository:
    """Load an offline Phase 14 staging-to-under-stand approach.

    The Phase 14 artifacts remain explicitly simulator-only candidates.  This
    loader verifies their static geometry gates; it does not turn route
    metadata into a certification claim.
    """

    def __init__(self, directory, candidates_file=None, source_files=None):
        self.directory = Path(directory)
        self.candidates_file = Path(candidates_file) if candidates_file else None
        self.source_files = source_files or {}

    def approach(self, station, entry_pose=None):
        filename = f'HOME__TO__{station}__REFINED.json'
        if not (self.directory / filename).exists():
            # Stops such as HOME have only per-gate candidates, no Phase 14 file.
            if entry_pose is None:
                raise ValueError(f'{station}: an entrance is needed to choose its approach')
            return self._entrance_approach(station, entry_pose)
        route = json.loads((self.directory / filename).read_text(encoding='utf-8'))
        validation = route.get('validation', {})
        if route.get('schema') != 'amr_phase14_two_segment_refined_v1':
            raise ValueError(f'Unsupported precision-docking artifact: {station}')
        if route.get('destination') != station:
            raise ValueError(f'Precision-docking identity mismatch: {station}')
        if validation.get('2d_leg_geometry') != 'PASSED':
            raise ValueError(f'Precision-docking leg geometry is not valid: {station}')
        vertical = validation.get('static_vertical_body_lift_clearance')
        if vertical not in ('PASSED', 'NOT_APPLICABLE_NO_TABLETOP_COLLISION'):
            raise ValueError(f'Precision-docking vertical clearance is not valid: {station}')

        poses = route.get('complete_poses')
        staging = route.get('staging_pose')
        if not isinstance(poses, list) or not isinstance(staging, list):
            raise ValueError(f'Precision-docking poses are missing: {station}')
        converted = [[float(value) for value in pose] for pose in poses]
        if any(len(pose) != 3 or not all(math.isfinite(value) for value in pose)
               for pose in converted):
            raise ValueError(f'Precision-docking pose data is invalid: {station}')
        start = min(range(len(converted)), key=lambda index: math.hypot(
            converted[index][0] - float(staging[0]),
            converted[index][1] - float(staging[1])))
        if math.hypot(converted[start][0] - float(staging[0]),
                      converted[start][1] - float(staging[1])) > 0.02:
            raise ValueError(f'Precision-docking staging handoff is invalid: {station}')
        approach = converted[start:]
        if len(approach) < 2:
            raise ValueError(f'Precision-docking approach is too short: {station}')
        if entry_pose is not None and math.hypot(
                approach[0][0] - float(entry_pose[0]),
                approach[0][1] - float(entry_pose[1])) > 0.03:
            return self._entrance_approach(station, entry_pose)
        return route, approach

    def _entrance_approach(self, station, entry_pose):
        if self.candidates_file is None:
            raise ValueError('No matching entrance candidate is available')
        data = json.loads(self.candidates_file.read_text(encoding='utf-8'))
        if data.get('schema') != 'mission_entrance_candidates_v1' or data.get('failures'):
            raise ValueError('Entrance candidate bundle is invalid')
        for key in ('world', 'robot', 'database'):
            path = self.source_files.get(key)
            if path is None or hashlib.sha256(Path(path).read_bytes()).hexdigest() != data.get('source_sha256', {}).get(key):
                raise ValueError(f'Entrance candidate source changed: {key}')
        choices = []
        for candidate in data.get('candidates', {}).values():
            if candidate.get('destination') != station:
                continue
            if (candidate.get('validation', {}).get('2d_leg_geometry') not in
                    ('PASSED', 'NOT_APPLICABLE_NO_LEGS') or
                    not candidate.get('static_2_5d_sampled_audit', {}).get('passed')):
                continue
            poses = candidate.get('complete_poses')
            if not isinstance(poses, list) or len(poses) < 2:
                continue
            converted = [[float(value) for value in pose] for pose in poses]
            if any(len(pose) != 3 or not all(math.isfinite(value) for value in pose)
                   for pose in converted):
                continue
            gap = math.hypot(converted[0][0] - float(entry_pose[0]),
                             converted[0][1] - float(entry_pose[1]))
            if gap <= 0.03:
                choices.append((gap, candidate, converted))
        if not choices:
            raise ValueError('No statically checked precision candidate matches this entrance')
        _, candidate, poses = min(choices, key=lambda item: item[0])
        return candidate, poses
