import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_route_database_v1 import constrained_astar


def test_clearance_bias_prefers_the_middle_of_a_free_corridor():
    safe = np.ones((9, 13), dtype=bool)
    clearance = np.ones_like(safe, dtype=float)
    clearance[4, :] = 4.0

    path = constrained_astar(
        (1, 2), (11, 6), safe, terminal_direction=0,
        clearance_map=clearance, clearance_weight=2.0)

    assert any(y == 4 for _, y in path)
