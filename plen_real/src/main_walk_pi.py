#!/usr/bin/env python
# IMPORTS
import numpy as np
from plen_real.servo_model import ServoJoint
# from plen_real.socket_comms import SocketClient, SocketServer
from plen_real.imu import IMU
import time


class PlenReal:
    def __init__(self):

        print("Initializing PLEN")

        # self.socket = Socket()

        print("Socket Ready!")

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
            [-0.95, 0.75],  # r_thigh_r_knee
            [-0.9, 0.3],  # r_knee_r_shin
            [-0.95, 1.2],  # r_shin_r_ankle
            [-0.8, 0.4],  # r_ankle_r_foot
            [-1.57, 1.57],  # LEFT LEG lb_servo_l_hip
            [-1.5, 0.15],  # l_hip_l_thigh
            [-0.75, 0.95],  # l_thigh_l_knee
            [-0.3, 0.9],  # l_knee_l_shin
            [-1.2, 0.95],  # l_shin_l_ankle
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
            False,  # LEFT ARM torso_l_shoulder
            False  # l_shoulder_ls_servo
            # False  # le_servo_l_elbow
        ]

        self.joint_list = []

        for i in range(len(self.joint_names)):
            if i < 8:
                # Use ADC 1, gpio 22
                self.joint_list.append(
                    ServoJoint(name=self.joint_names[i],
                               gpio=22,
                               fb_chan=i,
                               pwm_chan=i))
            elif i <= 12:
                # Use ADC 2, gpio 27
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

        joint_calib = input("Calibrate Joints [c] or Load Calibration [l] or Do Nothing [n]?")

        if joint_calib == "c":
            self.calibrate_motors()
        elif joint_calib == "l":
            for joint in self.joint_list:
                joint.load_calibration()

        print("Joints Ready!")

        self.torso_z = 0
        self.torso_y = 0
        self.torso_roll = 0
        self.torso_pitch = 0
        self.torso_yaw = 0
        self.torso_vx = 0

        input("PRESS ENTER TO CALIBRATE IMU")
        self.imu = IMU()

        print("PLEN READY TO GO!")

        self.time = time.time()

    def calibrate_motors(self):
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

    def step(self, action):
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
        # print(len(left_contact))

        self.left_contact = 0
        self.right_contact = 0

        # POSITION AND VELOCITY
        socket_msg = self.socket.receive_message()
        self.torso_y = socket_msg[1]
        self.torso_z = socket_msg[2]
        curr_time = time.time()
        self.torso_vx = (self.torso_x - socket_msg[0]) / float(curr_time -
                                                               self.time)
        self.torso_x = socket_msg[0]

        self.time = time.time()

        # ORIENTATION
        # Filter IMU
        self.imu.filter_rpy()
        # Read IMU
        self.imu.read_imu()
        self.torso_roll = self.imu.roll
        self.torso_pitch = self.imu.pitch
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
        """ Replays best policy directly from sim at 60Hz
    	"""
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

        # WILL USE ONE OF THE ABOVE, DEPENDS ON BEST PERFORMANCE
        choice = input("Use Command [c] or Position [p] trajectories?")

        loop_time = 1 / 60.0

        if choice == "c":
            # Use Actions
            for i in range(len(self.joint_cmds[0])):
                # Record current time
                start_time = time.time()
                for j in range(len(self.joint_cmds)):
                    joint_command = self.joint_cmds[j][i]
                    if not self.sim_to_real_key[i]:
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
                    self.joint_list[j].actuate(joint_command)
                elapsed_time = time.time() - start_time
                if loop_time > elapsed_time:
                    # Ensure 60Hz loop
                    time.sleep(loop_time - elapsed_time)


if __name__ == "__main__":
    plen = PlenReal()
