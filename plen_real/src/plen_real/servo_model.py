#!/usr/bin/env python

import numpy as np
import busio
import digitalio
import board
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn

import time
from adafruit_servokit import ServoKit


class ServoJoint:
    def __init__(self,
                 name,
                 effort=0.15,
                 speed=8.76,
                 gpio=22,
                 fb_chan=0,
                 pwm_chan=0,
                 pwm_min=600,
                 pwm_max=3000):
        # RIGHT LEG:
        # Joint 1 name: rb_servo_r_hip
        # Joint 2 name: r_hip_r_thigh
        # Joint 3 name: r_thigh_r_knee
        # Joint 4 name: r_knee_r_shin
        # Joint 5 name: r_shin_r_ankle
        # Joint 6 name: r_ankle_r_foot

        # LEFT LEG:
        # Joint 7 name: lb_servo_l_hip
        # Joint 8 name: l_hip_l_thigh
        # Joint 9 name: l_thigh_l_knee
        # Joint 10 name: l_knee_l_shin
        # Joint 11 name: l_shin_l_ankle
        # Joint 12 name: l_ankle_l_foot

        # RIGHT ARM:
        # Joint 13 name: torso_r_shoulder
        # Joint 14 name: r_shoulder_rs_servo
        # Joint 15 name: re_servo_r_elbow

        # LEFT ARM:
        # Joint 16 name: torso_l_shoulder
        # Joint 17 name: l_shoulder_ls_servo
        # Joint 18 name: le_servo_l_elbow
        self.name = name
        self.effort = effort  # Nm
        self.speed = speed  # rad/s

        # create the spi bus
        self.spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)

        # create the cs (chip select)
        if gpio == 22:
            self.cs = digitalio.DigitalInOut(board.D22)
        elif gpio == 27:
            self.cs = digitalio.DigitalInOut(board.D27)

        # create the mcp object
        self.mcp = MCP.MCP3008(self.spi, self.cs)

        # fb_chan from 0 to 7 for each MCP ADC
        if fb_chan == 0:
            self.chan = AnalogIn(self.mcp, MCP.P0)
        elif fb_chan == 1:
            self.chan = AnalogIn(self.mcp, MCP.P1)
        elif fb_chan == 2:
            self.chan = AnalogIn(self.mcp, MCP.P2)
        elif fb_chan == 3:
            self.chan = AnalogIn(self.mcp, MCP.P3)
        elif fb_chan == 4:
            self.chan = AnalogIn(self.mcp, MCP.P4)
        elif fb_chan == 5:
            self.chan = AnalogIn(self.mcp, MCP.P5)
        elif fb_chan == 6:
            self.chan = AnalogIn(self.mcp, MCP.P6)
        elif fb_chan == 7:
            self.chan = AnalogIn(self.mcp, MCP.P7)

        self.kit = ServoKit(channels=16)
        self.pwm_chan = pwm_chan
        self.kit.servo[self.pwm_chan].set_pulse_width_range(pwm_min, pwm_max)

        self.bias = 90.0  # degrees

    def forward_propagate(self, current_pos, desired_pos, dt):
        """ Predict the new position of the actuated servo
            motor joint
        """
        pos_change = desired_pos - current_pos
        percent_of_pos_reached = (self.speed * dt) / np.abs(pos_change)
        # Cap at 100%
        if percent_of_pos_reached > 100.0:
            percent_of_pos_reached = 100.0
        return current_pos + (pos_change * percent_of_pos_reached)

    def load_calibration(self):
        return True

    def calibrate(self, min_val, max_val, increment):
        # Send to min value and record digital sig
        # Send to max value and record digital sig

        # OR INCREMENT TO GET MORE DATA
        commands = np.array([])
        measurements = np.array([])

        # Number of data points to collect
        num_iters = 1000

        for i in range(num_iters):
            commanded_value = -90.0 + (i * 90 / num_iters)
            commands = np.append(commands, commanded_value)
            self.actuate(commanded_value)
            time.sleep(
                0.1 /
                (60.0 /
                 (180.0 * num_iters)))  # according to rated speed 0.1sec/60deg
            measurements = np.append(measurements, self.chan.value)

        # Perform fit
        polynomial = 4
        self.fit = np.polyfit(commands, measurements, polynomial)

    def remap(self, value):
        # Use calibraton value to remap from Digital Sig to Angle
        p = np.poly1d(self.fit)

        return p(value)

    def measure(self):
        return self.remap(self.chan.value)

    def rad2deg(self, rad):
        return rad * 180.0 / np.pi

    def actuate(self, desired_pos):
        self.kit.servo[
            self.pwm_chan].angle = self.bias + self.rad2deg(desired_pos)
