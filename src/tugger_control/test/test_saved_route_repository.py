import json

from tugger_control.saved_route_repository import (
    PrecisionDockingRepository,
    SavedRouteRepository,
)


def test_reads_a_validated_route(tmp_path):
    route = {
        'source': 'HOME', 'target': 'station',
        'poses': [[0, 0, 0], [1, 0, 0]],
    }
    database = {
        'version': 'AMR_V1_FIVE_STATION_DATABASE',
        'routes': {'HOME__TO__station': {
            'source': 'HOME', 'target': 'station',
            'status': 'VALIDATED_STATIC_CANDIDATE',
            'route_file': 'route.json',
        }},
    }
    (tmp_path / 'route_database.json').write_text(json.dumps(database))
    (tmp_path / 'route.json').write_text(json.dumps(route))
    loaded, poses = SavedRouteRepository(tmp_path).route('HOME__TO__station')
    assert loaded['target'] == 'station'
    assert poses == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]


def test_reads_the_precise_staging_to_dock_section(tmp_path):
    route = {
        'schema': 'amr_phase14_two_segment_refined_v1',
        'destination': 'cargo_station_0',
        'staging_pose': [2, 0, 0],
        'complete_poses': [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
        'validation': {
            '2d_leg_geometry': 'PASSED',
            'static_vertical_body_lift_clearance': 'PASSED',
        },
    }
    (tmp_path / 'HOME__TO__cargo_station_0__REFINED.json').write_text(json.dumps(route))
    loaded, poses = PrecisionDockingRepository(tmp_path).approach('cargo_station_0')
    assert loaded['destination'] == 'cargo_station_0'
    assert poses == [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
