#!/usr/bin/env python

import numpy as np


class JointModel:

    def __init__(self, effort=0.15, speed=8.76):
        self.effort = effort  # Nm
        self.speed = speed  # rad/s

    def forward_propagate(self, current_pos, desired_pos, dt):
        """ Predict the new position of the actuated servo
            motor joint
        """
        pos_change = desired_pos - current_pos
        percent_of_pos_reached = np.abs(pos_change) / (self.speed * dt)
        return current_pos + (pos_change * percent_of_pos_reached)
