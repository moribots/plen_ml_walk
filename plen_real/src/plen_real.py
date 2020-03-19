#!/usr/bin/env python
# IMPORTS
import numpy as np
from plen_real.servo_model import ServoJoint
# from plen_real.socket_comms import SocketClient
from plen_real.imu import IMU
import time
import os
# from plen_ros_helpers.td3 import TD3Agent


class PlenReal:
    def __init__(self):

        print("----------------------------")
        print("Initializing PLEN")

        # self.socket = SocketClient()
        print("----------------------------")
        print("Socket Ready!")

        print("----------------------------")
        print("CALIBRATING JOINTS")

        self.joint_names = [
            'rb_servo_r_hip', 'r_hip_r_thigh', 'r_thigh_r_knee',
            'r_knee_r_shin', 'r_shin_r_ankle', 'r_ankle_r_foot',
            'lb_servo_l_hip', 'l_hip_l_thigh', 'l_thigh_l_knee',
            'l_knee_l_shin', 'l_shin_l_ankle', 'l_ankle_l_foot',
            'torso_r_shoulder', 'r_shoulder_rs_servo', 're_servo_r_elbow',
            'torso_l_shoulder', 'l_shoulder_ls_servo', 'le_servo_l_elbow'
        ]

        # REAL
        # NOTE: NOT USING HAND JOINTS SINCE NEED EXTRA PWM BOARD AND ADC
        self.env_ranges = [
            [-1.57, 1.57],  # RIGHT LEG rb_servo_r_hip
            [-0.15, 1.5],  # r_hip_r_thigh
            [-0.95, 1.2],  # r_thigh_r_knee
            [-1.1, 1.57],  # r_knee_r_shin
            [-0.95, 1.2],  # r_shin_r_ankle
            [-0.8, 0.4],  # r_ankle_r_foot
            [-1.57, 1.57],  # LEFT LEG lb_servo_l_hip
            [-1.5, 0.15],  # l_hip_l_thigh
            [-1.2, 0.95],  # l_thigh_l_knee
            [-1.1, 1.57],  # l_knee_l_shin
            [-1.2, 1.2],  # l_shin_l_ankle
            [-0.4, 0.8],  # l_ankle_l_foot
            [-1.57, 1.57],  # RIGHT ARM torso_r_shoulder
            [-0.15, 1.57],  # r_shoulder_rs_servo
            # [-0.2, 0.35],  # re_servo_r_elbow
            [-1.57, 1.57],  # LEFT ARM torso_l_shoulder
            [-0.15, 1.57]  # l_shoulder_ls_servo
            # [-0.2, 0.35]  # le_servo_l_elbow
        ]

        # KEY TO FLIP MIN/MAX IN ENV RANGES ABOVE
        # IF TRUE: KEEP SAME
        # IF FALSE: FLIP
        self.sim_to_real_key = [
            False,  # RIGHT LEG rb_servo_r_hip
            True,  # r_hip_r_thigh
            True,  # r_thigh_r_knee
            True,  # r_knee_r_shin
            True,  # r_shin_r_ankle
            True,  # r_ankle_r_foot
            False,  # LEFT LEG lb_servo_l_hip
            True,  # l_hip_l_thigh
            True,  # l_thigh_l_knee
            True,  # l_knee_l_shin
            True,  # l_shin_l_ankle
            True,  # l_ankle_l_foot
            True,  # RIGHT ARM torso_r_shoulder
            False,  # r_shoulder_rs_servo
            # False,  # re_servo_r_elbow
            True,  # LEFT ARM torso_l_shoulder
            False  # l_shoulder_ls_servo
            # False  # le_servo_l_elbow
        ]

        self.servo_horn_bias = [
            -1,  # RIGHT LEG rb_servo_r_hip
            1,  # r_hip_r_thigh
            -1,  # r_thigh_r_knee
            1,  # r_knee_r_shin
            1,  # r_shin_r_ankle
            1,  # r_ankle_r_foot
            -1,  # LEFT LEG lb_servo_l_hip
            -1,  # l_hip_l_thigh
            1,  # l_thigh_l_knee
            0,  # l_knee_l_shin
            -1,  # l_shin_l_ankle
            0,  # l_ankle_l_foot
        ]
        self.servo_horn_bias = [i * 0.157 for i in self.servo_horn_bias]

        self.joint_list = []

        for i in range(len(self.joint_names)):
            if i < 8:
                # Use ADC 1, gpio 22
                self.joint_list.append(
                    ServoJoint(name=self.joint_names[i],
                               gpio=22,
                               fb_chan=i,
                               pwm_chan=i,
                               servo_horn_bias=self.servo_horn_bias[i]))
            elif i <= 12:
                # Use ADC 2, gpio 27
                if i < 12:
                    self.joint_list.append(
                        ServoJoint(name=self.joint_names[i],
                                   gpio=27,
                                   fb_chan=i - 8,
                                   pwm_chan=i,
                                   servo_horn_bias=self.servo_horn_bias[i]))
                else:
                    self.joint_list.append(
                        ServoJoint(name=self.joint_names[i],
                                   gpio=27,
                                   fb_chan=i - 8,
                                   pwm_chan=i))

        # NOW INITIALIZE ARM JOINTS; SPECIAL CASE FOR ONE PWM BOARD
        self.joint_list.append(
            ServoJoint(name=self.joint_names[13],
                       gpio=27,
                       fb_chan=13 - 8,
                       pwm_chan=13))
        # torso_l_shoulder
        self.joint_list.append(
            ServoJoint(name=self.joint_names[15],
                       gpio=27,
                       fb_chan=14 - 8,
                       pwm_chan=14))
        # l_shoulder_ls_servo
        self.joint_list.append(
            ServoJoint(name=self.joint_names[16],
                       gpio=27,
                       fb_chan=15 - 8,
                       pwm_chan=15))
        joint_calib = input(
            "Calibrate Joints [c] or Load Calibration [l] or Do Nothing [n]?")

        if joint_calib == "c":
            self.calibrate_motors()
        elif joint_calib == "l":
            for joint in self.joint_list:
                joint.load_calibration()

        for joint in self.joint_list:
            joint.actuate(0.0)

        # Set arms to 30deg
        self.joint_list[15].actuate(0.5)
        self.joint_list[13].actuate(0.5)

        print("Joints Ready!")

        print("----------------------------")

        self.torso_z = 0
        self.torso_y = 0
        self.torso_roll = 0
        self.torso_pitch = 0
        self.torso_yaw = 0
        self.torso_vx = 0

        input("PRESS ENTER TO CALIBRATE IMU")
        self.imu = IMU()

        print("----------------------------")
        print("Setting up RL Environment")
        # Observation Values
        # JOINTS (see self.env_ranges)
        # Low values of Joint Position Space
        self.joints_low = []
        # High values of Joint Position Space
        self.joints_high = []
        for j_state in self.env_ranges:
            self.joints_low.append(j_state[0])
            self.joints_high.append(j_state[1])

        # JOINT EFFORT - NOTE: UNUSED SINCE SERVO CANNOT MEASURE
        self.joint_effort_low = [-0.15] * 18
        self.joint_effort_high = [0.15] * 18

        # TORSO HEIGHT (0, 0.25)
        self.torso_height_min = 0
        self.torso_height_max = 0.25

        # TORSO TWIST (x) (-inf, inf)
        self.torso_vx_min = -np.inf
        self.torso_vx_max = np.inf

        self.torso_w_roll_min = -np.inf
        self.torso_w_roll_max = np.inf

        self.torso_w_pitch_min = -np.inf
        self.torso_w_pitch_max = np.inf

        self.torso_w_yaw_min = -np.inf
        self.torso_w_yaw_max = np.inf

        # TORSO ROLL (-pi, pi)
        self.torso_roll_min = -np.pi
        self.torso_roll_max = np.pi

        # TORSO PITCH (-pi, pi)
        self.torso_pitch_min = -np.pi
        self.torso_pitch_max = np.pi

        # TORSO YAW (-pi, pi)
        self.torso_yaw_min = -np.pi
        self.torso_yaw_max = np.pi

        # TORSO DEVIATION FROM X AXIS (-inf, inf)
        self.torso_y_min = -np.inf
        self.torso_y_max = np.inf

        # RIGHT FOOT CONTACT (0, 1)
        self.rfs_min = 0
        self.rfs_max = 1

        # LEFT FOOT CONTACT (0, 1)
        self.lfs_min = 0
        self.lfs_max = 1

        obs_dim = np.append(
            self.joints_low,
            np.array([
                self.torso_height_min, self.torso_vx_min, self.torso_roll_min,
                self.torso_pitch_min, self.torso_yaw_min, self.torso_y_min,
                self.rfs_min, self.lfs_min
            ]))
        self.state_dim = len(obs_dim)
        print("Environment Set!")
        print("----------------------------")

        print("PLEN READY TO GO!")

        self.time = time.time()

    def calibrate_motors(self):
        """ Take a number of measurements from the feedback potentiometer at
            various known motor positions and fit a polynomial for future
            readings
        """
        input("Press Enter when PLEN is elevated to a safe position:")
        for i in range(len(self.joint_list)):
            print("Calibrating Joint: " + self.joint_list[i].name)
            min_val = self.env_ranges[i][0]
            max_val = self.env_ranges[i][0]
            if not self.sim_to_real_key[i]:
                # Inicates URDF Opposite of Real
                min_val = -min_val
                max_val = -max_val
            self.joint_list[i].calibrate(min_val, max_val)

    # self.joint_list[0].calibrate(self.env_ranges[0][0],
    #                              self.env_ranges[0][1])

    def reset(self):
        """ Ends the Episode, returns the robot state, and resets the robot
        """
        self.episode_num += 1
        self.moving_avg_counter += 1
        self.cumulated_episode_reward = 0
        self.episode_timestep = 0
        # Reset Gait Params
        self.gait_period_counter = 0
        self.double_support_preriod_counter = 0
        self.right_contact_counter = 0
        self.left_contact_counter = 0
        self.lhip_joint_angles = np.array([])
        self.rhip_joint_angles = np.array([])
        self.lknee_joint_angles = np.array([])
        self.rknee_joint_angles = np.array([])
        self.lankle_joint_angles = np.array([])
        self.rankle_joint_angles = np.array([])
        print("PICK PLEN OFF THE GROUND AND HOLD IT BY THE TORSO.\n")
        input("PRESS ENTER TO RESET PLEN'S JOINTS TO ZERO.")

        for joint in self.joint_list:
            joint.actuate(0.0)

        input("PUT PLEN ON THE GROUND, PRESS ENTER TO CALIBRATE IMU")
        self.imu.calibrate()

        self.torso_x = 0
        input("Press Enter to Start")

    def step(self, action):
        """ Performs one action either using the Policy,
            or using joint positions directly.

            Then, computes an observation of the state,
            as well as the reward for this action,
            and determines whether the state is terminal.
        """
        # Convert agent actions into real actions
        env_action = np.zeros(18)
        # print("MESS {}".format(env_action))

        for i in range(len(action)):
            # Convert action from [-1, 1] to real env values
            env_action[i] = self.agent_to_env(self.env_ranges[i], action[i])

        for j in range(len(self.joint_list)):
            self.joint_list[i].actuate(env_action[i])

        observation = self.compute_observation()
        done = self.compute_done()
        reward = 0
        # reward = self.compute_reward()
        self.cumulated_episode_reward += reward
        self.episode_timestep += 1
        self.total_timesteps += 1
        # Increment Gait Reward Counters
        self.gait_period_counter += 1

        return observation, reward, done, {}

    def compute_done(self):
        """
        Decide if episode is done based on the observations

            - Pitch is above or below pi/2
            - Roll is above or below pi/2
            - Height is below height thresh
            - y position (abs) is above y thresh
            - episode timesteps above limit
        """
        if self.torso_roll > np.abs(np.pi / 3.) or self.torso_pitch > np.abs(
                np.pi / 3.) or self.torso_z < 0.08 or self.torso_y > 1:
            done = True
        else:
            done = False
        return done

    def compute_observation(self):
        """ Reads the relevant state parameters from the sensors
        """
        # print(len(left_contact))

        self.left_contact = 0
        self.right_contact = 0

        # POSITION AND VELOCITY
        self.socket.send_message("hello server")
        socket_msg = self.socket.receive_message()
        self.torso_y = socket_msg[1]
        self.torso_z = socket_msg[2]
        curr_time = time.time()
        self.torso_vx = (self.torso_x - socket_msg[0]) / float(curr_time -
                                                               self.time)
        self.time = curr_time
        self.torso_x = socket_msg[0]

        # ORIENTATION
        # READ IMU
        self.imu.filter_rpy()
        self.torso_roll = self.imu.true_roll
        self.torso_pitch = self.imu.true_pitch
        self.torso_yaw = self.imu.yaw

        # JOINT STATES
        self.joint_poses = []
        for i in range(len(self.joint_list)):
            self.joint_poses = np.append(self.joint_poses,
                                         self.joint_list[i].measure())

        observations = np.append(
            self.joint_poses,
            np.array([
                self.torso_z, self.torso_vx, self.torso_roll, self.torso_pitch,
                self.torso_yaw, self.torso_y, self.right_contact,
                self.left_contact
            ]))

        # Populate Joint Angle Difference Arrays using current - prev
        if len(self.lhip_joint_angles) > 0:
            # thigh_knee in URDF
            self.lhip_joint_angle_diff = self.lhip_joint_angles[
                -1] - self.joint_poses[2]
            self.rhip_joint_angle_diff = self.rhip_joint_angles[
                -1] - self.joint_poses[8]
            # knee_shin in URDF
            self.lknee_joint_angle_diff = self.lknee_joint_angles[
                -1] - self.joint_poses[3]
            self.rknee_joint_angle_diff = self.rknee_joint_angles[
                -1] - self.joint_poses[9]
            # shin_ankle in URDF
            self.lankle_joint_angle_diff = self.lankle_joint_angles[
                -1] - self.joint_poses[4]
            self.rankle_joint_angle_diff = self.rankle_joint_angles[
                -1] - self.joint_poses[10]
            self.first_pass = False
        else:
            self.first_pass = True
            self.lhip_joint_angle_diff = 0
            self.rhip_joint_angle_diff = 0
            self.lknee_joint_angle_diff = 0
            self.rknee_joint_angle_diff = 0
            self.lankle_joint_angle_diff = 0
            self.rankle_joint_angle_diff = 0

        # Pupulate joint angle arrays for gait calc
        # thigh_knee in URDF
        self.lhip_joint_angles = np.append(self.lhip_joint_angles,
                                           self.joint_poses[2])
        self.rhip_joint_angles = np.append(self.rhip_joint_angles,
                                           self.joint_poses[8])
        # knee_shin in URDF
        self.lknee_joint_angles = np.append(self.lknee_joint_angles,
                                            self.joint_poses[3])
        self.rknee_joint_angles = np.append(self.rknee_joint_angles,
                                            self.joint_poses[9])
        # shin_ankle in URDF
        self.lankle_joint_angles = np.append(self.lankle_joint_angles,
                                             self.joint_poses[4])
        self.rankle_joint_angles = np.append(self.rankle_joint_angles,
                                             self.joint_poses[10])

        return observations

    def agent_to_env(self, env_range, agent_val):
        """ Convert an action from the Agent space ([-1, 1])
            to the Environment Space
        """
        # Convert using y = mx + b
        agent_range = [-1, 1]
        # m = (y1 - y2) / (x1 - x2)
        m = (env_range[1] - env_range[0]) / (agent_range[1] - agent_range[0])
        # b = y1 - mx1
        b = env_range[1] - (m * agent_range[1])
        env_val = m * agent_val + b

        # Make sure no out of bounds
        if env_val >= env_range[1]:
            env_val = env_range[1] - 0.001
            # print("Sampled Too High!")
        elif env_val <= env_range[0]:
            env_val = env_range[0] + 0.001
            # print("Sampled Too Low!")

        return env_val

    def replay(self):
        """ Replays best policy directly from sim trajectories
        # """
        input("ENTER TO LOAD JOINT TRAJECTORIES")
        for i in range(12):
            self.joint_list[i].actuate(0.0)

        # Set arms to 30deg
        self.joint_list[15].actuate(0.5)
        self.joint_list[13].actuate(0.5)
        print("Loading Joint Trajectories...")
        # First, load the joint angles for each joint across each timestep

        # RIGHT LEG
        rhip_traj = []
        rthigh_traj = []
        rknee_traj = []
        rshin_traj = []
        rankle_traj = []
        rfoot_traj = []
        # LEFT LEG
        lhip_traj = []
        lthigh_traj = []
        lknee_traj = []
        lshin_traj = []
        lankle_traj = []
        lfoot_traj = []
        # RIGHT ARM
        rshoulder_traj = []
        rarm_traj = []
        # relbow_traj = []
        # LEFT ARM
        lshoulder_traj = []
        larm_traj = []
        # lelbow_traj = []

        self.joint_trajectories = [
            rhip_traj, rthigh_traj, rknee_traj, rshin_traj, rankle_traj,
            rfoot_traj, lhip_traj, lthigh_traj, lknee_traj, lshin_traj,
            lankle_traj, lfoot_traj, rshoulder_traj, rarm_traj, lshoulder_traj,
            larm_traj
        ]

        # Next, load the joint commands for each joint across each timestep

        # RIGHT LEG
        rhip_cmd = []
        rthigh_cmd = []
        rknee_cmd = []
        rshin_cmd = []
        rankle_cmd = []
        rfoot_cmd = []
        # LEFT LEG
        lhip_cmd = []
        lthigh_cmd = []
        lknee_cmd = []
        lshin_cmd = []
        lankle_cmd = []
        lfoot_cmd = []
        # RIGHT ARM
        rshoulder_cmd = []
        rarm_cmd = []
        # relbow_cmd = []
        # LEFT ARM
        lshoulder_cmd = []
        larm_cmd = []
        # lelbow_cmd = []

        self.joint_cmds = [
            rhip_cmd, rthigh_cmd, rknee_cmd, rshin_cmd, rankle_cmd, rfoot_cmd,
            lhip_cmd, lthigh_cmd, lknee_cmd, lshin_cmd, lankle_cmd, lfoot_cmd,
            rshoulder_cmd, rarm_cmd, lshoulder_cmd, larm_cmd
        ]

        # Load Both - EXCLUDE HAND JOINTS
        for i in range(len(self.joint_list)):
            self.joint_trajectories[i] = np.load(self.joint_list[i].name +
                                                 "_traj.npy")
            self.joint_cmds[i] = np.load(self.joint_list[i].name + "_cmd.npy")

        # Bend Legs
        full_bend = np.load("bend_traj.npy")

        # Exclude Elbows
        bend_legs = []
        for i in range(13):
            bend_legs.append(full_bend[i])
        bend_legs.append(0.5)
        bend_legs.append(full_bend[15])
        bend_legs.append(0.5)

        # print("BEND LEG SIZE {}".format(len(bend_legs)))
        # print("BEND LEG COMMAND")
        # print(bend_legs)

        input("PRESS ENTER TO BEND LEGS")

        print("------------------------------")
        print("BENDING LEGS")
        print("------------------------------")

        for j in range(12):
            joint_command = bend_legs[j]
            if not self.sim_to_real_key[j]:
                # Key indicates that URDF oposite of reality
                joint_command = -joint_command
            # print("{} COMMAND: \t {}".format(self.joint_list[j].name,
            #                                  joint_command))
            self.joint_list[j].actuate(joint_command)

        print("LEGS BENT")
        print("------------------------------")


        # WILL USE ONE OF THE ABOVE, DEPENDS ON BEST PERFORMANCE
        choice = input("Use Command [c] or Position [p] trajectories?")

        loop_time = 1 / 20.0  # 1/Hz

        if choice == "c":
            # Use Actions
            for i in range(len(self.joint_cmds[0])):
                # Record current time
                start_time = time.time()
                for j in range(len(self.joint_cmds)):
                    joint_command = self.agent_to_env(self.env_ranges[j], self.joint_cmds[j][i])
                    if not self.sim_to_real_key[j]:
                        # Key indicates that URDF oposite of reality
                        joint_command = -joint_command
                    self.joint_list[j].actuate(joint_command)
                elapsed_time = time.time() - start_time
                if loop_time > elapsed_time:
                    # Ensure 60Hz loop
                    time.sleep(loop_time - elapsed_time)
        elif choice == "p":
            # Use Joint Positions
            for i in range(len(self.joint_trajectories[0])):
                # Record current time
                start_time = time.time()
                for j in range(len(self.joint_trajectories)):
                    joint_command = self.joint_trajectories[j][i]
                    if not self.sim_to_real_key[j]:
                        # Key indicates that URDF oposite of reality
                        joint_command = -joint_command
                    # print("{} COMMAND: \t {}".format(self.joint_list[j].name,
                    #                                  joint_command))
                    self.joint_list[j].actuate(joint_command)
                elapsed_time = time.time() - start_time
                if loop_time > elapsed_time:
                    # Ensure 60Hz loop
                    time.sleep(loop_time - elapsed_time)
                    print("RATE: {}".format(1 / (time.time() - start_time)))

        for i in range(12):
            self.joint_list[i].actuate(0.0)

        # Set arms to 30deg
        self.joint_list[15].actuate(0.5)
        self.joint_list[13].actuate(0.5)

    # def deploy(self):
    #     """ Deploy Live Policy using real sensors
    #     """
    #     # Find abs path to this file
    #     my_path = os.path.abspath(os.path.dirname(__file__))
    #     models_path = os.path.join(my_path, "../models")
    #     if not os.path.exists(models_path):
    #         os.makedirs(models_path)

    #     state_dim = self.state_dim
    #     action_dim = 18
    #     max_action = 1.0

    #     print("RECORDED MAX ACTION: {}".format(max_action))

    #     policy = TD3Agent(state_dim, action_dim, max_action)
    #     # Optionally load existing policy, replace 9999 with num
    #     policy_num = 3229999  # 629999 current best policy
    #     if os.path.exists(models_path + "/" + "plen_walk_gazebo_" +
    #                       str(policy_num) + "_critic"):
    #         print("Loading Existing Policy")
    #         policy.load(models_path + "/" + "plen_walk_gazebo_" +
    #                     str(policy_num))

    #     state = self.compute_observation
    #     done = False
    #     episode_reward = 0
    #     episode_timesteps = 0
    #     episode_num = 0
    #     t = 0  # total timesteps
    #     evaluations = []

    #     run = True

    #     while run:
    #         t += 1
    #         episode_timesteps += 1
    #         # Deterministic Policy Action
    #         action = np.clip(policy.select_action(np.array(state)),
    #                          -max_action, max_action)
    #         # rospy.logdebug("Selected Acton: {}".format(action))

    #         # Perform action
    #         next_state, reward, done, _ = self.step(action)

    #         state = next_state
    #         episode_reward += reward
    #         # print("DT REWARD: {}".format(reward))

    #         if done:
    #             # +1 to account for 0 indexing.
    #             # +0 on ep_timesteps since it will increment +1 even if done=True
    #             print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".
    #                   format(t + 1, episode_num, episode_timesteps,
    #                          episode_reward))
    #             # Reset environment
    #             reset_quit = input("Reset or Quit [q]?")
    #             if reset_quit == "q":
    #                 run = False
    #                 break
    #             state, done = self.reset(), False
    #             evaluations.append(episode_reward)
    #             episode_reward = 0
    #             episode_timesteps = 0
    #             episode_num += 1


if __name__ == "__main__":
    plen = PlenReal()
    for i in range(20):
        plen.replay()
