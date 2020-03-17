#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import math
from plen_bullet.plen_env import PlenWalkEnv


class TrajectoryGenerator():
    def __init__(self):
        self.l_hip_knee = 25.0
        self.l_knee_foot = 40.0
        self.env = PlenWalkEnv()

    def setup_trajectory_params(self,
                                num_DoubleSupport,
                                num_SingleSupport,
                                height,
                                stride,
                                sit,
                                swayBody,
                                foot_sway,
                                fwd_bias,
                                sway_steps,
                                lift_pushoff=0.4,
                                land_pullback=0.6,
                                timeStep=0.1):
        # the number of point when two feet are landed
        self.num_DoubleSupport = num_DoubleSupport
        # the number of point when lift one foot
        self.num_SingleSupport = num_SingleSupport
        # foot lift height
        self.foot_lift_height = height
        # stride length
        self._l = stride
        # sit height. increase this will make leg more fold. too high or too low makes an error
        self._sit = sit
        # body sway length
        self._swayBody = swayBody
        # foot sway length. 0 -> feet move straight forward. plus this make floating leg spread.(increase gap between feet)
        self._foot_sway = foot_sway
        # start point of sway
        self._sway_steps = sway_steps
        # push the lifting foot backward when lifting the foot to gains momentum.
        self._lift_pushoff = lift_pushoff
        # Before put the foot down, go forward more and pull back when landing.
        self._land_pullback = land_pullback
        # simulation timeStep
        self._timeStep = timeStep
        # Forward bias for torso
        self.fwd_bias = fwd_bias

        self._stepPoint = num_DoubleSupport + num_SingleSupport

    def zero_traj(self):
        # Rows: x,y,z | Columns: number of trajectory points
        # START AND END CONDITION
        self.Zero_Legs = np.zeros(
            (3, self.num_double_support + self.num_single_support))

    # def bend_knees(self):
    #     # Rows: x,y,z | Columns: number of trajectory points
    #     # START AND END CONDITION
    #     # TODO:

    def first_step(self):
        trajectoryLength = self._l * (
            2.0 * self.num_DoubleSupport + self.num_SingleSupport) / (
                self.num_DoubleSupport + self.num_SingleSupport)
        walkPoint = self.num_DoubleSupport * 2.0 + self.num_SingleSupport * 2.0
        # Rows: x,y,z | Columns: number of trajectory points
        # START AND END CONDITION
        self.foot_start_rfwd_r = np.zeros(
            (3, self.num_DoubleSupport + self.num_SingleSupport))
        self.foot_start_lfwd_r = np.zeros(
            (3, self.num_DoubleSupport + self.num_SingleSupport))

        for i in range(self.num_DoubleSupport - self._sway_steps):
            # SIT
            t = (i + 1.0) / self.num_DoubleSupport
            self.foot_start_rfwd_r[0][i] = 0
            self.foot_start_rfwd_r[2][i] = self._sit

            self.foot_start_lfwd_r[0][i] = 0
            self.foot_start_lfwd_r[2][i] = self._sit

        for i in range(self.num_SingleSupport):
            # MOVE LEG FORWARD
            t = (i + 1.0) / self.num_SingleSupport
            t2 = (i + 1.0) / (self.num_SingleSupport + self._sway_steps)
            sin_tpi = math.sin(t * math.pi)

            self.foot_start_rfwd_r[2][
                i + self.num_DoubleSupport - self._sway_steps] = math.sin(
                    t * math.pi) * self.foot_lift_height + self._sit
            # print("t: {}".format(t))
            self.foot_start_rfwd_r[0][
                i + self.num_DoubleSupport - self._sway_steps] = (
                    2 * t + (1 - t) * self._lift_pushoff * -sin_tpi +
                    t * self._land_pullback * sin_tpi) * trajectoryLength / 4

            self.foot_start_lfwd_r[0][
                i + self.num_DoubleSupport - self._sway_steps] = (
                    math.cos(t2 * math.pi / 2) - 1) * trajectoryLength * (
                        (self._sway_steps + self.num_DoubleSupport +
                         self.num_SingleSupport) /
                        (self.num_DoubleSupport * 2 + self.num_SingleSupport) -
                        0.5)

            self.foot_start_lfwd_r[2][i + self.num_DoubleSupport -
                                      self._sway_steps] = self._sit

        for i in range(self._sway_steps):
            # PUT FORWARD LEG DOWN
            t2 = (i + 1.0 + self.num_SingleSupport) / (self.num_SingleSupport +
                                                       self._sway_steps)

            self.foot_start_rfwd_r[0][
                i + self.num_SingleSupport + self.num_DoubleSupport -
                self._sway_steps] = -trajectoryLength * (
                    (i + 1) / (walkPoint - self.num_SingleSupport) - 0.5)
            self.foot_start_rfwd_r[2][i + self.num_SingleSupport +
                                      self.num_DoubleSupport -
                                      self._sway_steps] = self._sit

            self.foot_start_lfwd_r[0][
                i + self.num_SingleSupport + self.num_DoubleSupport -
                self._sway_steps] = (
                    math.cos(t2 * math.pi / 2) - 1) * trajectoryLength * (
                        (self._sway_steps + self.num_DoubleSupport +
                         self.num_SingleSupport) /
                        (self.num_DoubleSupport * 2 + self.num_SingleSupport) -
                        0.5)
            self.foot_start_lfwd_r[2][i + self.num_SingleSupport +
                                      self.num_DoubleSupport -
                                      self._sway_steps] = self._sit

        for i in range(self.num_DoubleSupport + self.num_SingleSupport):
            # Y CONTROL
            t = (i + 1.0) / (self.num_DoubleSupport + self.num_SingleSupport)
            if t < 1.0 / 4.0:
                self.foot_start_rfwd_r[1][i] = -self._swayBody * (
                    math.sin(t * math.pi) - (1 - math.sin(math.pi * 2 * t)) *
                    (math.sin(4 * t * math.pi) / 4))
                self.foot_start_lfwd_r[1][i] = self._swayBody * (
                    math.sin(t * math.pi) - (1 - math.sin(math.pi * 2 * t)) *
                    (math.sin(4 * t * math.pi) / 4))
            else:
                self.foot_start_rfwd_r[1][i] = -self._swayBody * math.sin(
                    t * math.pi)
                self.foot_start_lfwd_r[1][i] = self._swayBody * math.sin(
                    t * math.pi)

    def intermediate_steps(self):
        walkPoint = self.num_DoubleSupport * 2.0 + self.num_SingleSupport * 2.0
        trajectoryLength = self._l * (
            2.0 * self.num_DoubleSupport + self.num_SingleSupport) / (
                self.num_DoubleSupport + self.num_SingleSupport)

        walkPoint0 = np.zeros((3, self.num_DoubleSupport))
        walkPoint1 = np.zeros((3, self.num_SingleSupport))
        walkPoint2 = np.zeros((3, self.num_DoubleSupport))
        walkPoint3 = np.zeros((3, self.num_SingleSupport))

        # walking motion
        for i in range(self.num_DoubleSupport):
            t = (i + 1.0) / (walkPoint - self.num_SingleSupport)
            walkPoint0[0][i] = -trajectoryLength * (t - 0.5)
            walkPoint0[2][i] = self._sit
            walkPoint0[1][i] = self._swayBody * math.sin(2 * math.pi * (
                (i + 1 - self._sway_steps) / walkPoint))

        for i in range(self.num_SingleSupport):
            t = (i + 1.0 + self.num_DoubleSupport) / (walkPoint -
                                                      self.num_SingleSupport)
            walkPoint1[0][i] = -trajectoryLength * (t - 0.5)
            walkPoint1[2][i] = self._sit
            walkPoint1[1][i] = self._swayBody * math.sin(2 * math.pi * (
                (i + 1 + self.num_DoubleSupport - self._sway_steps) /
                walkPoint))

        for i in range(self.num_DoubleSupport):
            t = (i + 1.0 + self.num_DoubleSupport +
                 self.num_SingleSupport) / (walkPoint - self.num_SingleSupport)
            walkPoint2[0][i] = -trajectoryLength * (t - 0.5)
            walkPoint2[2][i] = self._sit
            walkPoint2[1][i] = self._swayBody * math.sin(2 * math.pi * (
                (i + 1 + self.num_DoubleSupport + self.num_SingleSupport -
                 self._sway_steps) / walkPoint))

        for i in range(self.num_SingleSupport):
            t = (i + 1.0) / self.num_SingleSupport
            sin_tpi = math.sin(t * math.pi)

            walkPoint3[0][i] = (
                2 * t - 1 + (1 - t) * self._lift_pushoff * -sin_tpi +
                t * self._land_pullback * sin_tpi) * trajectoryLength / 2
            walkPoint3[2][i] = math.sin(
                t * math.pi) * self.foot_lift_height + self._sit
            walkPoint3[1][i] = math.sin(
                t * math.pi) * self._foot_sway + self._swayBody * math.sin(
                    2 * math.pi *
                    ((i + 1 + walkPoint - self.num_SingleSupport -
                      self._sway_steps) / walkPoint))

        # Forward Bias
        if self.fwd_bias != 0:
            walkPoint0[0] = walkPoint0[0] - self.fwd_bias
            walkPoint1[0] = walkPoint1[0] - self.fwd_bias
            walkPoint2[0] = walkPoint2[0] - self.fwd_bias
            walkPoint3[0] = walkPoint3[0] - self.fwd_bias

        self.foot_walk_lfwd_r = np.column_stack([
            walkPoint0[:, self._sway_steps:], walkPoint1,
            walkPoint2[:, :self._sway_steps]
        ])
        self.foot_walk_rfwd_r = np.column_stack([
            walkPoint2[:, self._sway_steps:], walkPoint3,
            walkPoint0[:, :self._sway_steps]
        ])

    def last_step(self):
        trajectoryLength = self._l * (
            2.0 * self.num_DoubleSupport + self.num_SingleSupport) / (
                self.num_DoubleSupport + self.num_SingleSupport)
        walkPoint = self.num_DoubleSupport * 2.0 + self.num_SingleSupport * 2.0
        self.foot_end_rfwd_r = np.zeros(
            (3, self.num_DoubleSupport + self.num_SingleSupport))
        self.foot_end_lfwd_r = np.zeros(
            (3, self.num_DoubleSupport + self.num_SingleSupport))

        for i in range(self.num_DoubleSupport - self._sway_steps):
            self.foot_end_lfwd_r[0][i] = -trajectoryLength * \
                ((i+1+self._sway_steps)/(walkPoint-self.num_SingleSupport)-0.5)
            self.foot_end_lfwd_r[2][i] = self._sit

            self.foot_end_rfwd_r[0][i] = -trajectoryLength * \
                ((i + 1 + self._sway_steps + self.num_DoubleSupport+self.num_SingleSupport)/(walkPoint-self.num_SingleSupport)-0.5)
            self.foot_end_rfwd_r[2][i] = self._sit
        for i in range(self.num_SingleSupport):
            t = (i + 1.0) / self.num_SingleSupport
            sin_tpi = math.sin(t * math.pi)

            self.foot_end_lfwd_r[0][
                i + self.num_DoubleSupport - self._sway_steps] = (
                    math.sin(t * math.pi / 2) - 1) * trajectoryLength * (
                        (self.num_DoubleSupport) /
                        (self.num_DoubleSupport * 2 + self.num_SingleSupport) -
                        0.5)
            self.foot_end_lfwd_r[2][i + self.num_DoubleSupport -
                                    self._sway_steps] = self._sit

            self.foot_end_rfwd_r[0][
                i + self.num_DoubleSupport - self._sway_steps] = (
                    2 * t - 2 + (1 - t) * self._lift_pushoff * -sin_tpi +
                    t * self._land_pullback * sin_tpi) * trajectoryLength / 4
            self.foot_end_rfwd_r[2][
                i + self.num_DoubleSupport - self._sway_steps] = math.sin(
                    t * math.pi) * self.foot_lift_height + self._sit
        for i in range(self._sway_steps):
            self.foot_end_lfwd_r[0][i + self.num_DoubleSupport +
                                    self.num_SingleSupport -
                                    self._sway_steps] = 0
            self.foot_end_lfwd_r[2][i + self.num_DoubleSupport +
                                    self.num_SingleSupport -
                                    self._sway_steps] = self._sit

            self.foot_end_rfwd_r[0][i + self.num_DoubleSupport +
                                    self.num_SingleSupport -
                                    self._sway_steps] = 0
            self.foot_end_rfwd_r[2][i + self.num_DoubleSupport +
                                    self.num_SingleSupport -
                                    self._sway_steps] = self._sit

        # Forward Bias
        if self.fwd_bias != 0:
            self.foot_start_rfwd_r[
                0] = self.foot_start_rfwd_r[0] - self.fwd_bias
            self.foot_start_lfwd_r[
                0] = self.foot_start_lfwd_r[0] - self.fwd_bias
            self.foot_end_lfwd_r[0] = self.foot_end_lfwd_r[0] - self.fwd_bias
            self.foot_end_rfwd_r[0] = self.foot_end_rfwd_r[0] - self.fwd_bias

    def assemble_trajectories(self):
        self.foot_walk_lfwd_l = self.foot_walk_rfwd_r * np.array([[1], [-1],
                                                                  [1]])
        self.foot_walk_rfwd_l = self.foot_walk_lfwd_r * np.array([[1], [-1],
                                                                  [1]])

        self.foot_start_rfwd_l = self.foot_start_lfwd_r * np.array([[1], [-1],
                                                                    [1]])
        self.foot_start_lfwd_l = self.foot_start_rfwd_r * np.array([[1], [-1],
                                                                    [1]])

        self.foot_end_lfwd_l = self.foot_end_rfwd_r * np.array([[1], [-1], [1]
                                                                ])
        self.foot_end_rfwd_l = self.foot_end_lfwd_r * np.array([[1], [-1], [1]
                                                                ])

    def inverseKinematicsList(self, point, isRightLeg):
        inverseAngle = np.zeros((point[0].size, 6))
        # print("========================================")
        for i in range(point[0].size):
            # print("-----------------------------------")
            lhip_knee = self.l_hip_knee
            lknee_foot = self.l_knee_foot

            fx = point[0][i]
            fy = point[1][i]
            fz = self.l_hip_knee + self.l_knee_foot - point[2][i]
            # print("fx: {}".format(fx))
            # print("fy: {}".format(fy))
            # print("fz: {}".format(fz))

            # R = math.sqrt(fx**2 + fz**2)

            # th4 = math.atan2(
            #     math.sqrt(1 - ((R**2 - lknee_foot**2 - lhip_knee**2) /
            #                    (2.0 * (lhip_knee * lknee_foot)))**2),
            #     (R**2 - lknee_foot**2 - lhip_knee**2) /
            #     (2.0 * (lhip_knee * lknee_foot)))

            # ta = math.atan2(fx, fz)
            # tb = math.atan2(lhip_knee * math.sin(th4),
            #                 lknee_foot + lhip_knee * math.cos(th4))

            # th5 = ta + tb

            # th3 = -th4 - th5

            if isRightLeg:
                th1 = math.atan2(-fy, fz)
            else:
                th1 = math.atan2(fy, fz)

            th3 = math.acos((fx**2 + fy**2 + fz**2 - lhip_knee**2 - lknee_foot**2) / (2.0*lhip_knee*lknee_foot))

            sqrtyz = math.sqrt(fy**2 + fz**2)
            hok = lhip_knee / lknee_foot

            th2 = -math.atan2((sqrtyz * math.sin(th3) + fx*math.cos(th3) + fx*hok) , (sqrtyz * math.cos(th3) + sqrtyz*hok -fx*math.sin(th3)))

            # NOTE: KNEE [3] or [9] IS CAPPED AT +-[-1.1, 0.3] RADIANS FOR MAX AND THIGH [2] or [8] IS CAPPED AT +- [-0.95, 0.75] RADIANS

            if isRightLeg:
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

            if isRightLeg:
                th5 = -th1
            else:
                th5 = th1

            inverseAngle[i] = np.array([0, th1, th2, th3, th4, th5])

        return inverseAngle

    def joint_space_trajectories(self):
        # # START
        # RIGHT SIDE START
        self.foot_start_rfwd = np.column_stack([
            # RIGHT COMPONENT OF RIGHT SIDE START
            self.inverseKinematicsList(self.foot_start_rfwd_r, True),
            # LEFT COMPONENT OF RIGHT SIDE START
            self.inverseKinematicsList(self.foot_start_rfwd_l, False)
        ])
        # LEFT SIDE START
        self.foot_start_lfwd = np.column_stack([
            # RIGHT COMPONENT OF LEFT SIDE START
            self.inverseKinematicsList(self.foot_start_lfwd_r, True),
            # LEFT COMPONENT OF LEFT SIDE START
            self.inverseKinematicsList(self.foot_start_lfwd_l, False)
        ])

        # # END
        # RIGHT SIDE END
        self.foot_end_rfwd = np.column_stack([
            # RIGHT COMPONENT OF RIGHT SIDE END
            self.inverseKinematicsList(self.foot_end_rfwd_r, True),
            # LEFT COMPONENT OF RIGHT SIDE END
            self.inverseKinematicsList(self.foot_end_rfwd_l, False)
        ])

        # LEFT SIDE END
        self.foot_end_lfwd = np.column_stack([
            # RIGHT COMPONENT OF LEFT SIDE END
            self.inverseKinematicsList(self.foot_end_lfwd_r, True),
            # LEFT COMPONENT OF LEFT SIDE END
            self.inverseKinematicsList(self.foot_end_lfwd_l, False)
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
        self.setup_trajectory_params(num_DoubleSupport=10,
                                     num_SingleSupport=10,
                                     height=30.0,
                                     stride=10.0,
                                     sit=10.0,
                                     swayBody=5.0,
                                     foot_sway=-0.0,
                                     fwd_bias=15.0,
                                     sway_steps=5,
                                     lift_pushoff=0.0,
                                     land_pullback=4.0,
                                     timeStep=0.0165)

        # Starting Motion
        self.first_step()

        # Walking Motion
        self.intermediate_steps()

        # Ending Motion
        self.last_step()

        # Assemble

        self.assemble_trajectories()

        # Convert to joint space
        self.joint_space_trajectories()


if __name__ == "__main__":
    traj = TrajectoryGenerator()

    traj.main()