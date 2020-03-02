#!/usr/bin/env python

import time
from adafruit_servokit import ServoKit
kit = ServoKit(channels=16)
"""
https://learn.adafruit.com/adafruit-16-channel-pwm-servo-hat-for-raspberry-pi/using-the-python-library
"""

calib = input("Calibration [c] or Test [t]?: ")

loop = True

while loop:

    if calib == "c":
        channel = int(
            input("Which channel is your servo connected to? [0-15]: "))
        kit.servo[channel].set_pulse_width_range(600, 3000)

        print("Setting Servo to 90 degrees: 0 for action space of [-1, 1]: ")
        kit.servo[channel].angle = 90.0

        cont = input("Calibrate another motor [y] or quit [n]? ")

        if cont == "n":
            loop = False

    elif calib == "t":
        channel = int(
            input("Which channel is your servo connected to? [0-15]: "))
        kit.servo[channel].set_pulse_width_range(600, 3000)

        val = float(input("Select a HIGH angle value (deg): "))
        kit.servo[channel].angle = val + 90.0
        val = float(input("Select a LOW angle value (deg): "))
        kit.servo[channel].angle = val + 90.0
        input("Press Enter to send servo to 0 (90): ")
        kit.servo[channel].angle = 90.0

        cont = input("Test another motor [y] or quit [n]? ")

        if cont == "n":
            loop = False
