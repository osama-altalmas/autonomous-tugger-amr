import pytest

from tugger_control.cargo_policy import check_destination, confirmed_cargo


def test_confirmed_cargo_requires_both_fresh_state_reports():
    assert confirmed_cargo({0: 'detached', 3: 'detached'}, {0: 0.1, 3: 0.1}) is None
    assert confirmed_cargo({0: 'attached', 3: 'detached'}, {0: 0.1, 3: 0.1}) == 0
    assert confirmed_cargo({0: 'detached', 3: 'attached'}, {0: 0.1, 3: 0.1}) == 3
    with pytest.raises(ValueError, match='stale'):
        confirmed_cargo({0: 'attached', 3: 'detached'}, {0: 0.1, 3: 3.0})
    with pytest.raises(ValueError, match='Multiple'):
        confirmed_cargo({0: 'attached', 3: 'attached'}, {0: 0.1, 3: 0.1})


def test_pickup_and_dropoff_destinations_use_fixed_cargo_pairs():
    check_destination('cargo_station_0', None)
    check_destination('cargo_station_3', None)
    check_destination('cargo_dropoff_northwest', 0)
    check_destination('cargo_dropoff_southeast', 3)
    check_destination('HOME', 0)
    with pytest.raises(ValueError, match='does not match'):
        check_destination('cargo_dropoff_southeast', 0)
    with pytest.raises(ValueError, match='does not match'):
        check_destination('cargo_dropoff_northwest', 3)
    with pytest.raises(ValueError, match='unloaded'):
        check_destination('cargo_station_3', 0)
