#!/usr/bin/env python
import numpy as np
import time


# Class for object detection
class Detector():
    def __init__(self, position, num_particles):
        self.position = position  # x, y, theta at world origin
        # Number of particles
        self.num_part = num_particles
        # linear velocity noise
        self.var_v = 0.003
        # angular velocity noise
        self.var_w = 0.003
        # positional heading noise due to linear velocity
        self.noise_1 = 0.01
        # positional heading noise due to angular velocity
        self.noise_2 = 0.01
        # empty numpy array w/N rows, 4 columns for x,y,theta, weight
        self.particles = np.zeros((self.num_part, 4))
        # set initial particle weights (4th col) to be all equal and sum to 1
        self.particles[:, 3] = 1.0 / self.num_part

    def noisy_controls(self, controls):
        return True

