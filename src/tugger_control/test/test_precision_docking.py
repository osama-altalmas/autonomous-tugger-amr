from tugger_control.precision_docking import final_approach_tail


def test_final_approach_retries_the_remaining_clearance_checked_poses():
    poses = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
             [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    index, tail = final_approach_tail(poses, 2.08, 0.01)
    assert index == 2
    assert tail == [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]


def test_final_approach_keeps_two_poses_when_robot_is_nearest_the_goal():
    poses = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    index, tail = final_approach_tail(poses, 2.04, 0.0)
    assert index == 2
    assert tail == [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
