#!/usr/bin/env python

import numpy as np
import math
import matplotlib.pyplot as plt
from plen_bullet.plen_env import PlenWalkEnv


class TrajectoryGenerator():
    def __init__(self,
                 num_DoubleSupport=5,
                 num_SingleSupport=10,
                 height=30.0,
                 stride=30.0,
                 bend_distance=10.0,
                 body_sway=5.0,
                 fwd_bias=10.0,
                 sway_steps=5):
        """ Initialize trajectory generation parameters.
            This only has the ability to implement a forward
            trajectory, and with no consideration for feedback
            control.

            ZMP with LQI is a good candidate for future improvement.

            Note that regardless of the solver use, the foot trajectory
            must provide a solution to the IK problem.

            see: https://www.hindawi.com/journals/mpe/2015/437979/
            for details

            NOTE: All Values in mm
        """
        # Length of hip-knee segment
        self.l_hip_knee = 25.0
        # Length of knee-ankle segment
        self.l_knee_foot = 40.0
        # Num. Timesteps spent with both feet planted
        self.num_DoubleSupport = num_DoubleSupport
        # Num. Timesteps spent in stride per trajectory
        self.num_SingleSupport = num_SingleSupport
        # Foot height during stride
        self.foot_lift_height = height
        # stride length
        self.stride_length = stride
        # Distance to bend down
        self._bend_distance = bend_distance
        # body sway length
        self._body_sway = body_sway
        # Forward bias for torso
        self.fwd_bias = fwd_bias
        self.env = PlenWalkEnv(joint_act=True)

    def foot_path(self):
        """ Generate Foot trajectories for Dominant and Support in
            cartesian coordinates relative to their respective hips.

            # sinewave trajectory generation loosely based on
            # https://github.com/Einsbon/bipedal-robot-walking-simulation/blob/master/walkGenerator.py
        """
        DS_support_foot = np.zeros((3, 2 * self.num_DoubleSupport))
        SS_support_foot = np.zeros((3, self.num_SingleSupport))
        DS_dominant_foot = np.zeros((3, 2 * self.num_DoubleSupport))
        SS_dominant_foot = np.zeros((3, self.num_SingleSupport))

        # Dominant Foot - SS
        for i in range(self.num_SingleSupport):
            # t FROM 0 to 1
            t = i / (self.num_SingleSupport - 1.0)
            # Linear Forward Trajectory: t * stride_length
            SS_dominant_foot[0][i] = t * self.stride_length
            # Clipped Sinewave Trajectory for Sway from 1/3 to 2/3 of -pi
            # DS will handle 0 to 1/3 and 2/3 to 1
            SS_dominant_foot[1][i] = np.sin(-np.pi *
                                            ((1 / 3.0) *
                                             (1 + t))) * self._body_sway
            # Sinewave Trajectory for Height from 0 to pi
            SS_dominant_foot[2][i] = np.sin(
                t * np.pi) * self.foot_lift_height + self._bend_distance

        # Dominant Foot - DS
        # We do twice the trajectory to use half for each foot (see below)
        for i in range(2 * self.num_DoubleSupport):
            # t FROM 0 to 1
            t = i / (2 * self.num_DoubleSupport - 1.0)

            # Move by 1/4 stride length for each foot
            DS_dominant_foot[0][i] = -self.stride_length * (t / 2.0)
            # Clipped Sinewave Trajectory for Sway from -1/3 to 1/3 of -pi
            # 0 to 1/3 for dominant support (1st half)
            # -1/3 to 0 for support support (2nd half)
            DS_dominant_foot[1][i] = np.sin(-np.pi *
                                            ((-1.0 / 3.0) +
                                             (2 / 3.0) * t)) * self._body_sway
            DS_dominant_foot[2][i] = self._bend_distance

        # Support Foot - SS
        for i in range(self.num_SingleSupport):
            # t FROM 0 to 1
            t = i / (self.num_SingleSupport - 1.0)
            # moving from 1/4 stride length to -1/4 stride length in this segment
            SS_support_foot[0][i] = self.stride_length * (
                (1.0 / 2.0) - t) / 2.0

            # Clipped Sinewave Trajectory for Sway from 1/3 to 2/3 of pi
            # DS will handle 0 to 1/3 and 2/3 to 1
            SS_support_foot[1][i] = np.sin(np.pi * ((1 / 3.0) *
                                                    (1 + t))) * self._body_sway
            SS_support_foot[2][i] = self._bend_distance

        # Support Foot - DS
        # We do twice the trajectory to use half for each foot (see below)
        for i in range(2 * self.num_DoubleSupport):
            # t FROM 0 to 1
            t = i / (2.0 * self.num_DoubleSupport - 1.0)

            # Move by 1/4 stride length for each foot
            DS_support_foot[0][i] = self.stride_length * (1.0 - t) / 2.0

            # Clipped Sinewave Trajectory for Sway from 2/3 to 4/3 of pi
            # 2/3 to 1 for support support (1st half)
            # 1 to 4/3 for dominant support (2nd half)
            DS_support_foot[1][i] = np.sin(-np.pi *
                                           ((2.0 / 3.0) +
                                            (2 / 3.0) * t)) * self._body_sway
            DS_support_foot[2][i] = self._bend_distance

        # Forward Bias
        # Subtract foor position by amount of forward tilt
        if self.fwd_bias != 0:
            DS_support_foot[0] = DS_support_foot[0] - self.fwd_bias
            SS_support_foot[0] = SS_support_foot[0] - self.fwd_bias
            DS_dominant_foot[0] = DS_dominant_foot[0] - self.fwd_bias
            SS_dominant_foot[0] = SS_dominant_foot[0] - self.fwd_bias

        # Dominant foot does DS(1st half), Single Support, DS(2nd half)
        self.foot_walk_rfwd_r = np.column_stack([
            DS_dominant_foot[:, self.num_DoubleSupport:], SS_dominant_foot,
            DS_support_foot[:, :self.num_DoubleSupport]
        ])

        # Support foot does: DS(1st half, Lift, DS(2nd half)
        self.foot_walk_lfwd_r = np.column_stack([
            DS_support_foot[:, self.num_DoubleSupport:], SS_support_foot,
            DS_dominant_foot[:, :self.num_DoubleSupport]
        ])

        # store for plot
        self.SS_dominant_foot = SS_dominant_foot
        self.DS_dominant_foot = DS_dominant_foot
        self.SS_support_foot = SS_support_foot
        self.DS_support_foot = DS_support_foot

    def assemble_trajectories(self):
        """ Assemble complements of trajectories in self.foot_path
            to change leading foot.

            Note that for example, if the right foot is the dominant one,
            the left foot has the same trajectory as the right foot were it
            a support foot, but with its y position negated
        """
        # Right and left foot trajectories are the same,
        # except that the y component is flipped
        self.foot_walk_lfwd_l = self.foot_walk_rfwd_r * np.array([[1], [-1],
                                                                  [1]])
        self.foot_walk_rfwd_l = self.foot_walk_lfwd_r * np.array([[1], [-1],
                                                                  [1]])

    def IK(self, point, RightLeg):
        """ Inverse Kinematics to actuate foot with respect to hip joint
        """
        # Joint Angles for each leg
        joint_angles = np.zeros((point[0].size, 6))
        # SOURCE: https://www.hindawi.com/journals/mpe/2015/437979/
        for i in range(point[0].size):
            lhip_knee = self.l_hip_knee
            lknee_foot = self.l_knee_foot

            # Zx, Zy, Zz relative to hip joint
            Zx = point[0][i]
            Zy = point[1][i]
            Zz = self.l_hip_knee + self.l_knee_foot - point[2][i]

            if RightLeg:
                th1 = math.atan2(-Zy, Zz)
            else:
                th1 = math.atan2(Zy, Zz)

            th3 = math.acos(
                (Zx**2 + Zy**2 + Zz**2 - lhip_knee**2 - lknee_foot**2) /
                (2.0 * lhip_knee * lknee_foot))

            sqrtyz = np.sqrt(Zy**2 + Zz**2)
            hok = lhip_knee / lknee_foot

            th2 = -math.atan2(
                (sqrtyz * np.sin(th3) + Zx * np.cos(th3) + Zx * hok),
                (sqrtyz * np.cos(th3) + sqrtyz * hok - Zx * np.sin(th3)))

            # NOTE: Cap Angles based on actual joint limits
            if RightLeg:
                # Check [3]knee and [2]thigh
                if th3 < self.env.real_ranges[3][0]:
                    th3 = self.env.real_ranges[3][0]
                elif th3 > self.env.real_ranges[3][1]:
                    th3 = self.env.real_ranges[3][1]

                if th2 < self.env.real_ranges[2][0]:
                    th2 = self.env.real_ranges[2][0]
                elif th2 > self.env.real_ranges[2][1]:
                    th2 = self.env.real_ranges[2][1]
            else:
                # Check [9]knee and [8]thigh
                if th3 < self.env.real_ranges[9][0]:
                    th3 = self.env.real_ranges[9][0]
                elif th3 > self.env.real_ranges[9][1]:
                    th3 = self.env.real_ranges[9][1]

                if th2 < self.env.real_ranges[8][0]:
                    th2 = self.env.real_ranges[8][0]
                elif th2 > self.env.real_ranges[8][1]:
                    th2 = self.env.real_ranges[8][1]

            th4 = -(th2 + th3)

            if RightLeg:
                th5 = -th1
            else:
                th5 = th1

            joint_angles[i] = np.array([0, th1, th2, th3, th4, th5])

        return joint_angles

    def joint_space_trajectories(self):
        """ Assemble joint-space trajectories from EE-space trajectories
            fed through IK process
        """
        # Amount to bend down
        # can incrememnt bend as robot moves down for smooth transistion
        # Rows: Trajectory | Columns: x,y,z of foot wrt hip
        bend_array = np.column_stack([
            np.array([0.0, 0.0, self._bend_distance]),
            np.array([0.0, 0.0, self._bend_distance]),
            np.array([0.0, 0.0, self._bend_distance])
        ])
        # # BEND LEGS
        self.bend = np.column_stack([
            # RIGHT COMPONENT OF RIGHT SIDE START
            self.IK(bend_array, True),
            # LEFT COMPONENT OF RIGHT SIDE START
            self.IK(bend_array, False)
        ])

        # # WALK
        # RIGHT SIDE WALK
        self.foot_walk_rfwd = np.column_stack([
            # RIGHT COMPONENT OF RIGHT SIDE WALK
            self.IK(self.foot_walk_rfwd_r, True),
            # LEFT COMPONENT OF RIGHT SIDE WALK
            self.IK(self.foot_walk_rfwd_l, False)
        ])

        # LEFT SIDE WALK
        self.foot_walk_lfwd = np.column_stack([
            # RIGHT COMPONENT OF LEFT SIDE WALK
            self.IK(self.foot_walk_lfwd_r, True),
            # LEFT COMPONENT OF LEFT SIDE WALK
            self.IK(self.foot_walk_lfwd_l, False)
        ])

    def main(self):
        # Walking Motion
        self.foot_path()

        # Assemble
        self.assemble_trajectories()

        # Convert to joint space
        self.joint_space_trajectories()

    def plot(self):
        """ Visualize trajctories for portfolio
        """
        # Walking Motion
        self.foot_path()

        # Assemble
        self.assemble_trajectories()

        # DOMINANT FOOT
        # X,Y,Z TRAJ
        plt.figure(0)
        plt.autoscale(enable=True, axis='both', tight=None)
        plt.title('Dominant Foot Trajectories - SS')
        plt.ylabel('positon (mm)')
        plt.xlabel('timestep')
        plt.plot(self.SS_dominant_foot[0], color='r', label="x")
        plt.plot(self.SS_dominant_foot[1], color='g', label="y")
        plt.plot(self.SS_dominant_foot[2], color='b', label="z")
        plt.yticks(np.arange(-20.0, 45.0 + 1.0, 2.0))

        plt.figure(1)
        plt.autoscale(enable=True, axis='both', tight=None)
        plt.title('Dominant Foot Trajectories - DS')
        plt.ylabel('positon (mm)')
        plt.xlabel('timestep')
        plt.plot(self.DS_dominant_foot[0], color='r', label="x")
        plt.plot(self.DS_dominant_foot[1], color='g', label="y")
        plt.plot(self.DS_dominant_foot[2], color='b', label="z")

        # SUPPORT FOOT
        # X,Y,Z TRAJ
        plt.figure(2)
        plt.autoscale(enable=True, axis='both', tight=None)
        plt.title('Support Foot Trajectories - SS')
        plt.ylabel('positon (mm)')
        plt.xlabel('timestep')
        plt.plot(self.SS_support_foot[0], color='r', label="x")
        plt.plot(self.SS_support_foot[1], color='g', label="y")
        plt.plot(self.SS_support_foot[2], color='b', label="z")

        plt.figure(3)
        plt.autoscale(enable=True, axis='both', tight=None)
        plt.title('Support Foot Trajectories - DS')
        plt.ylabel('positon (mm)')
        plt.xlabel('timestep')
        plt.plot(self.DS_support_foot[0], color='r', label="x")
        plt.plot(self.DS_support_foot[1], color='g', label="y")
        plt.plot(self.DS_support_foot[2], color='b', label="z")

        plt.legend()
        plt.show()


if __name__ == '__main__':
    traj = TrajectoryGenerator()
    traj.plot()
