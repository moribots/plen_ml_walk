#!/usr/bin/env python

import time
from adafruit_servokit import ServoKit
kit = ServoKit(channels=16)

"""
https://learn.adafruit.com/adafruit-16-channel-pwm-servo-hat-for-raspberry-pi/using-the-python-library
"""

flag = True

# Modify servo on channel 0 (tunable param)
# mod pw range to get 180deg rotation
# if servo whines, then range too high
kit.servo[0].set_pulse_width_range(600, 3000)

for i in range(1000):
    if flag:
        kit.servo[0].angle = 180
        flag = False
    else:
        kit.servo[0].angle = 0
        flag = True
    time.sleep(0.3)
