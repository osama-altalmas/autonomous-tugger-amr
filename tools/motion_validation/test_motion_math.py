import math
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src/tugger_control'))
from tugger_control.motion_math import approach, steering_for_twist, speed_limit_for_steering, wheel_speeds

assert approach(0, 1, .25, .1) == .025
assert abs(approach(.1, 0, .35, .1) - .065) < 1e-10
assert abs(steering_for_twist(.1, .1, .58, math.pi/4)-math.atan(.29)) < 1e-9
assert abs(steering_for_twist(-.1, .1, .58, math.pi/4)+math.atan(.29)) < 1e-9
assert speed_limit_for_steering(math.pi/4,.58,.05,.2) < .2
assert abs(wheel_speeds(.1,0,.58,.44,.055)[2]) < 1e-12
assert wheel_speeds(-.1,-.3,.58,.44,.055)[2] > 0
print('PASS: ramp, braking, reverse steering, lateral speed cap, traction kinematics')
