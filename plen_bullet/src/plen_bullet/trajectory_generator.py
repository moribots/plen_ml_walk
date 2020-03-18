#!/usr/bin/env python

import numpy as np
import math
from plen_bullet.plen_env import PlenWalkEnv


class TrajectoryGenerator():
    def __init__(self,
                 num_DoubleSupport=10,
                 num_SingleSupport=10,
                 height=30.0,
                 stride=30.0,
                 bend_distance=10.0,
                 body_sway=5.0,
                 foot_sway=-0.0,
                 fwd_bias=15.0,
                 sway_steps=5):
        self.l_hip_knee = 25.0
        self.l_knee_foot = 40.0
        self.num_DoubleSupport = num_DoubleSupport
        self.num_SingleSupport = num_SingleSupport
        # Foot height during stride
        self.foot_lift_height = height
        # stride length
        self.stride_length = stride
        # Distance to bend down
        self._bend_distance = bend_distance
        # body sway length
        self._body_sway = body_sway
        # number of steps dedicating to swaying (moving torso to the side)
        self._sway_steps = sway_steps
        # Forward bias for torso
        self.fwd_bias = fwd_bias
        self._stepPoint = num_DoubleSupport + num_SingleSupport
        self.env = PlenWalkEnv()

    def foot_path(self):
        DS_dominant_foot = np.zeros((3, self.num_DoubleSupport))
        SS_dominant_foot = np.zeros((3, self.num_SingleSupport))
        DS_support_foot = np.zeros((3, self.num_DoubleSupport))
        SS_support_foot = np.zeros((3, self.num_SingleSupport))

        # Double the period so that each segment of the trajectory has its dts
        Period = 2.0 * (self.num_DoubleSupport + self.num_SingleSupport)

        # Support Foot - SS
        for i in range(self.num_SingleSupport):
            t = (i + 1.0) / self.num_SingleSupport
            # Linear Forward Trajectory: t * stride_length
            SS_support_foot[0][i] = t * self.stride_length
            # Clipped Sinewave Trajectory for Sway
            SS_support_foot[1][i] = self._body_sway * np.sin(2.0 * np.pi * (
                (i + 1.0 + Period - self.num_SingleSupport - self._sway_steps)
                / Period))
            # Clipped Sinewave Trajectory for Height
            SS_support_foot[2][i] = np.sin(
                t * np.pi) * self.foot_lift_height + self._bend_distance
        # Dominant Foot - DS
        for i in range(self.num_DoubleSupport):
            # Subtract Single Support from total Period
            t = (i + 1.0) / (Period - self.num_SingleSupport)
            DS_dominant_foot[0][i] = (0.5 - t) * self.stride_length
            DS_dominant_foot[1][i] = self._body_sway * np.sin(2 * np.pi * (
                (i + 1.0 - self._sway_steps) / Period))
            DS_dominant_foot[2][i] = self._bend_distance

        # Dominant Foot - SS
        for i in range(self.num_SingleSupport):
            # Add Elapsed Double Support and subtract Single Support
            t = (i + 1.0 + self.num_DoubleSupport) / (Period -
                                                      self.num_SingleSupport)
            SS_dominant_foot[0][i] = (0.5 - t) * self.stride_length
            SS_dominant_foot[1][i] = self._body_sway * np.sin(2.0 * np.pi * (
                (i + 1.0 + self.num_DoubleSupport - self._sway_steps) /
                Period))
            SS_dominant_foot[2][i] = self._bend_distance
        # Support Foot - DS
        for i in range(self.num_DoubleSupport):
            # Add Elapsed Double and Single Supports and sub Single Support
            t = (i + 1.0 + self.num_DoubleSupport +
                 self.num_SingleSupport) / (Period - self.num_SingleSupport)
            DS_support_foot[0][i] = (0.5 - t) * self.stride_length
            DS_support_foot[2][i] = self._bend_distance
            DS_support_foot[1][i] = self._body_sway * np.sin(2.0 * np.pi * (
                (i + 1.0 + self.num_DoubleSupport + self.num_SingleSupport -
                 self._sway_steps) / Period))

        # Forward Bias
        if self.fwd_bias != 0:
            DS_dominant_foot[0] = DS_dominant_foot[0] - self.fwd_bias
            SS_dominant_foot[0] = SS_dominant_foot[0] - self.fwd_bias
            DS_support_foot[0] = DS_support_foot[0] - self.fwd_bias
            SS_support_foot[0] = SS_support_foot[0] - self.fwd_bias

        # Dominant foot does: DS, Lift, DS
        self.foot_walk_lfwd_r = np.column_stack([
            DS_dominant_foot[:, self._sway_steps:], SS_dominant_foot,
            DS_support_foot[:, :self._sway_steps]
        ])

        # Support foot does DS, Single Support, DS
        self.foot_walk_rfwd_r = np.column_stack([
            DS_support_foot[:, self._sway_steps:], SS_support_foot,
            DS_dominant_foot[:, :self._sway_steps]
        ])

    def assemble_trajectories(self):
        # Right and left foot trajectories are the same,
        # except that the y component is flipped
        self.foot_walk_lfwd_l = self.foot_walk_rfwd_r * np.array([[1], [-1],
                                                                  [1]])
        self.foot_walk_rfwd_l = self.foot_walk_lfwd_r * np.array([[1], [-1],
                                                                  [1]])

    def inverseKinematicsList(self, point, RightLeg):
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
                if th3 < self.env.env_ranges[3][0]:
                    th3 = self.env.env_ranges[3][0]
                elif th3 > self.env.env_ranges[3][1]:
                    th3 = self.env.env_ranges[3][1]

                if th2 < self.env.env_ranges[2][0]:
                    th2 = self.env.env_ranges[2][0]
                elif th2 > self.env.env_ranges[2][1]:
                    th2 = self.env.env_ranges[2][1]
            else:
                # Check [9]knee and [8]thigh
                if th3 < self.env.env_ranges[9][0]:
                    th3 = self.env.env_ranges[9][0]
                elif th3 > self.env.env_ranges[9][1]:
                    th3 = self.env.env_ranges[9][1]

                if th2 < self.env.env_ranges[8][0]:
                    th2 = self.env.env_ranges[8][0]
                elif th2 > self.env.env_ranges[8][1]:
                    th2 = self.env.env_ranges[8][1]

            th4 = -(th2 + th3)

            if RightLeg:
                th5 = -th1
            else:
                th5 = th1

            joint_angles[i] = np.array([0, th1, th2, th3, th4, th5])

        return joint_angles

    def joint_space_trajectories(self):
        # Amount to bend down
        # can incrememnt bend as robot moves down for smooth transistion
        bend_array = np.column_stack([
            np.array([0.0, 0.0, self._bend_distance]),
            np.array([0.0, 0.0, self._bend_distance]),
            np.array([0.0, 0.0, self._bend_distance])
        ])
        # # BEND LEGS
        self.bend = np.column_stack([
            # RIGHT COMPONENT OF RIGHT SIDE START
            self.inverseKinematicsList(bend_array, True),
            # LEFT COMPONENT OF RIGHT SIDE START
            self.inverseKinematicsList(bend_array, False)
        ])

        # # WALK
        # RIGHT SIDE WALK
        self.foot_walk_rfwd = np.column_stack([
            # RIGHT COMPONENT OF RIGHT SIDE WALK
            self.inverseKinematicsList(self.foot_walk_rfwd_r, True),
            # LEFT COMPONENT OF RIGHT SIDE WALK
            self.inverseKinematicsList(self.foot_walk_rfwd_l, False)
        ])

        # LEFT SIDE WALK
        self.foot_walk_lfwd = np.column_stack([
            # RIGHT COMPONENT OF LEFT SIDE WALK
            self.inverseKinematicsList(self.foot_walk_lfwd_r, True),
            # LEFT COMPONENT OF LEFT SIDE WALK
            self.inverseKinematicsList(self.foot_walk_lfwd_l, False)
        ])

    def main(self):
        # Walking Motion
        self.foot_path()

        # Assemble
        self.assemble_trajectories()

        # Convert to joint space
        self.joint_space_trajectories()


if __name__ == "__main__":
    traj = TrajectoryGenerator()

    traj.main()