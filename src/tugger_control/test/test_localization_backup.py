"""The localization fallback: AMCL's own map copy, re-centring and the watchdog."""
import math
import re
from pathlib import Path

import numpy as np
import yaml

from tugger_control.refined_localization import (continues, CONTINUITY_SHIFT_M,
                                                 DISAGREE_SHIFT_M, disagrees,
                                                 recentre_covariance, RECENTRE_GOOD_FITS,
                                                 RECENTRE_INTERVAL_S, RECENTRE_POSITION_STD_M,
                                                 RECENTRE_SHIFT_M, should_recentre)

ROOT = Path(__file__).resolve().parents[3]
MAPS = ROOT / 'src/tugger_navigation/maps'


def test_amcl_map_is_the_localization_map_half_a_cell_on():
    true_map = yaml.safe_load((MAPS / 'localization_map.yaml').read_text())
    amcl_map = yaml.safe_load((MAPS / 'localization_map_amcl.yaml').read_text())
    assert amcl_map['image'] == true_map['image']
    assert amcl_map['resolution'] == true_map['resolution']
    half = true_map['resolution'] / 2
    for axis in (0, 1):
        assert math.isclose(amcl_map['origin'][axis], true_map['origin'][axis] + half)


def test_amcl_reads_its_own_copy_and_the_launch_serves_it():
    amcl = (ROOT / 'src/tugger_navigation/config/amcl.yaml').read_text()
    assert re.search(r'map_topic: /localization_map_amcl\b', amcl)
    assert re.search(r'tf_broadcast: false', amcl)
    launch = (ROOT / 'src/tugger_navigation/launch/localization.launch.py').read_text()
    assert "'topic_name': 'localization_map_amcl'" in launch
    assert "'amcl_map_server'" in launch and 'refined_localization' in launch


def test_nav2_rounding_on_the_shifted_map_lands_on_the_true_cell():
    # nav2_amcl: i = floor((x - origin) / res + 0.5). With the origin half a
    # cell on, every point maps to the cell that truly contains it.
    true_map = yaml.safe_load((MAPS / 'localization_map.yaml').read_text())
    res, ox = true_map['resolution'], true_map['origin'][0]
    shifted = ox + res / 2
    for x in (ox + 0.001, ox + 0.0199, ox + 3.137, ox + 17.99):
        assert math.floor((x - shifted) / res + 0.5) == math.floor((x - ox) / res)


def test_recentre_only_after_a_run_of_good_fully_pinned_fits():
    big = RECENTRE_SHIFT_M + 0.01
    assert should_recentre(RECENTRE_GOOD_FITS, 3, big, 0.0, None, False)
    assert not should_recentre(RECENTRE_GOOD_FITS - 1, 3, big, 0.0, None, False)
    assert should_recentre(RECENTRE_GOOD_FITS, 2, big, 0.0, None, False)       # aisle
    assert not should_recentre(RECENTRE_GOOD_FITS, 1, big, 0.0, None, False)


def test_recentre_needs_real_drift_and_respects_interval_and_zones():
    n = RECENTRE_GOOD_FITS
    assert not should_recentre(n, 3, RECENTRE_SHIFT_M - 0.005, 0.0, None, False)
    assert should_recentre(n, 3, 0.0, math.radians(0.6), None, False)          # heading drift
    assert not should_recentre(n, 3, 0.05, 0.0, RECENTRE_INTERVAL_S - 0.1, False)
    assert should_recentre(n, 3, 0.05, 0.0, RECENTRE_INTERVAL_S, False)
    assert not should_recentre(n, 3, 0.05, 0.0, None, True)                    # station zone


def test_watchdog_threshold():
    assert not disagrees(DISAGREE_SHIFT_M - 0.01, 0.0)
    assert disagrees(DISAGREE_SHIFT_M + 0.01, 0.0)
    assert disagrees(0.0, math.radians(3.5))


def test_an_aisle_recentre_keeps_amcl_spread_along_the_aisle():
    amcl = [0.0] * 36
    amcl[0], amcl[7], amcl[35] = 0.15 ** 2, 0.04 ** 2, math.radians(2) ** 2
    cov = recentre_covariance([(1.0, 0.0, 0.0)], amcl)      # along x not pinned
    assert math.isclose(cov[0][0], 0.15 ** 2)
    assert math.isclose(cov[1][1], RECENTRE_POSITION_STD_M ** 2)
    assert cov[2][2] < math.radians(1) ** 2
    full = recentre_covariance([], amcl)
    assert np.allclose(np.diag(full)[:2], RECENTRE_POSITION_STD_M ** 2)


def test_a_fit_far_from_amcl_is_used_only_when_it_continues_the_last_fit():
    robot = (3.0, 1.0, 0.2)
    last = (0.50, -0.20, 0.01)
    near = (0.50 + CONTINUITY_SHIFT_M / 2, -0.20, 0.01)
    assert continues(last, near, robot, 0.5, 0.18)
    assert not continues(last, (0.70, -0.20, 0.01), robot, 0.5, 0.18)    # a jump
    assert not continues(last, near, robot, 5.0, 0.18)                    # too long ago
    assert not continues(None, near, robot, 0.5, 0.18)
    assert not continues(last, near, robot, 0.5, 0.80)                    # AMCL far gone


def test_a_search_result_is_judged_same_relocalize_or_lost():
    from tugger_control.refined_localization import judge_search
    assert judge_search(0.30, 0.50, 0.95, 0.03, 0.01) == 'same'          # clutter only
    assert judge_search(0.20, 0.70, 0.60, 0.65, 0.17) == 'relocalize'    # pushed by hand
    assert judge_search(0.20, 0.70, 0.95, 0.65, 0.17) == 'lost'          # not unique
    assert judge_search(0.40, 0.50, 0.60, 0.65, 0.17) == 'lost'          # not clearly better
    assert judge_search(0.10, 0.20, 0.50, 0.03, 0.01) == 'lost'          # nothing fits
