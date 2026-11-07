"""Station exit from a dock pose slightly beside the approach line."""

import math

from tugger_control.station_departure import plan_station_departure

# North exit: staging (0, 2) to dock (0, 0), 1 cm spacing.
APPROACH = [(0.0, 2.0 - 0.01 * i, -math.pi / 2) for i in range(201)]


def test_on_the_line_the_exit_is_the_reversed_approach():
    poses = plan_station_departure((0.0, 0.0, -math.pi / 2), APPROACH)['poses']
    assert all(abs(x) < 1e-9 for x, _, _ in poses)


def test_a_few_centimetres_beside_the_line_the_exit_blends_onto_it():
    # After a turn in place the centre drifted 3.3 cm sideways (Gazebo).
    departure = plan_station_departure((0.033, 0.012, -1.581), APPROACH)
    poses = departure['poses']
    assert poses[0][:2] == [0.033, 0.012]
    assert all(abs(x) < 1e-9 for x, y, _ in poses if y > 0.5)
    lateral = [x for x, _, _ in poses]
    assert all(b <= a + 1e-12 for a, b in zip(lateral, lateral[1:]))
