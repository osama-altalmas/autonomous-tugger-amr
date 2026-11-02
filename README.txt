Motion control batch: offline-tested command acceleration/deceleration, steering slew,
lateral acceleration speed cap, immediate traction watchdog stop, diagnostic topic.

Extract archive in workspace root; run bash INSTALL_AND_VALIDATE.sh.
The installer backs up original controller to four_ws_controller.py.before_smoothing.
No changes to nav2.yaml, odometry, robot model, or FollowPath executor.

After installation restart the tugger_control launch to load new controller.
New /four_ws/motion_state Float64MultiArray:
[target_v, commanded_v, target_front_delta, commanded_front_delta, commanded_omega]

Important: steering slew causes a temporary yaw mismatch with the Nav2 requested twist.
Watch tracking error; do not assume this is fully validated from offline math tests.
Emergency watchdog stops traction immediately and holds steering angle.
Current limits are conservative for the 0.2 m/s RPP setting.
