import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register

import pybullet as p
import pybullet_data
import numpy as np

import time

register(
    id="PlenWalkEnv-v1",
    entry_point='plen_bullet.plen_env:PlenWalkEnv',
    max_episode_steps=300,
)


class PlenWalkEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __init__(self, render=True):
        super(PlenWalkEnv, self).__init__()

        self.running_step = 1. / 60.
        self.timestep = 1. / 240.
        self.sim_stepsize = int(self.running_step / self.timestep)
        print("--------------------------------------------")
        print("SIM STEP SIZE: {}".format(self.sim_stepsize))
        print("--------------------------------------------")

        # Learning Info Loggers
        self.episode_num = 0
        self.cumulated_episode_reward = 0
        self.episode_timestep = 0
        self.total_timesteps = 0
        # Set up values for moving average pub
        self.moving_avg_buffer_size = 1000
        self.moving_avg_buffer = np.zeros(self.moving_avg_buffer_size)
        self.moving_avg_counter = 0

        # Possible Rewards
        self.reward_range = (-np.inf, np.inf)

        self.max_episode_steps = 300

        # Reward for being alive
        self.dead_penalty = 100.
        self.alive_reward = self.dead_penalty / self.max_episode_steps
        # Reward for forward velocity
        self.vel_weight = 3.
        # Reward for maintaining original height
        self.init_height = 0.158
        self.height_weight = 20.
        # Reward for staying on x axis
        self.straight_weight = 1
        # Reward staying upright
        self.roll_weight = 1.
        # Reward for staying upright
        self.pitch_weight = 0.5
        # reward for facing forward
        self.yaw_weight = 1.
        # Reward for minimal joint actuation
        self.joint_effort_weight = 0.035
        # Whether the episode is done due to failure
        self.dead = False

        # Agent Action Space
        low_act = np.ones(18) * -1
        high_act = np.ones(18)
        self.action_space = spaces.Box(low_act, high_act, dtype=np.float32)

        # Environment Action Space
        # self.env_ranges = [
        #     [-1.7, 1.7],  # RIGHT LEG
        #     [-1.54, 0.12],
        #     [-1.7, 0.75],
        #     [-0.2, 0.95],
        #     [-0.95, 1.54],
        #     [-0.45, 0.8],
        #     [-1.7, 1.7],  # LEFT LEG
        #     [-0.12, 1.54],
        #     [-0.75, 1.7],
        #     [-0.95, 0.2],
        #     [-1.54, 0.95],
        #     [-0.8, 0.45],
        #     [-1.7, 1.7],  # RIGHT ARM
        #     [-0.15, 1.7],
        #     [-0.2, 0.5],
        #     [-1.7, 1.7],  # LEFT ARM
        #     [-0.15, 1.7],
        #     [-0.2, 0.5]
        # ]
        self.env_ranges = [
            [-1.57, 1.57],  # RIGHT LEG
            [-0.15, 1.5],
            [-0.95, 0.75],
            [-0.9, 0.3],
            [-0.95, 1.2],
            [-0.8, 0.4],
            [-1.57, 1.57],  # LEFT LEG
            [-1.5, 0.15],
            [-0.75, 0.95],
            [-0.3, 0.9],
            [-1.2, 0.95],
            [-0.4, 0.8],
            [-1.57, 1.57],  # RIGHT ARM
            [-0.15, 1.57],
            [-0.2, 0.35],
            [-1.57, 1.57],  # LEFT ARM
            [-0.15, 1.57],
            [-0.2, 0.35]
        ]

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

        obs_low = np.append(
            self.joints_low,
            np.array([
                self.torso_height_min, self.torso_vx_min, self.torso_roll_min,
                self.torso_pitch_min, self.torso_yaw_min, self.torso_y_min,
                self.rfs_min, self.lfs_min
            ]))

        obs_high = np.append(
            self.joints_high,
            np.array([
                self.torso_height_max, self.torso_vx_max, self.torso_roll_max,
                self.torso_pitch_max, self.torso_yaw_max, self.torso_y_max,
                self.rfs_max, self.lfs_max
            ]))
        self.observation_space = spaces.Box(obs_low, obs_high)

        self.torso_z = 0
        self.torso_y = 0
        self.torso_roll = 0
        self.torso_pitch = 0
        self.torso_yaw = 0
        self.torso_vx = 0
        self.torso_w_roll = 0
        self.torso_w_pitch = 0
        self.torso_w_yaw = 0

        if (render):
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)  # non-graphical version
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath())  # used by loadURDF
        p.resetDebugVisualizerCamera(cameraDistance=0.8,
                                     cameraYaw=45,
                                     cameraPitch=-30,
                                     cameraTargetPosition=[0, 0, 0])
        self._seed()
        p.setRealTimeSimulation(0)  # 1=Realtime, 0=Simtime
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)  # m/s^2
        self.frame_skip = 4
        self.numSolverIterations = 5
        # p.setPhysicsEngineParameter(
        #     fixedTimeStep=self.timestep * self.frame_skip,
        #     numSolverIterations=self.numSolverIterations,
        #     numSubSteps=self.frame_skip)
        # p.setTimeStep(0.01)   # sec
        # p.setTimeStep(0.001)  # sec
        self.plane = p.loadURDF("plane.urdf")
        self.StartPos = [0, 0, 0.158]
        self.StartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robotId = p.loadURDF("plen.urdf", self.StartPos,
                                  self.StartOrientation)
        # Gathered from experiment, see self.move_joints
        self.movingJoints = [
            5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21, 24, 26, 27, 30
        ]

        # Get the number of joints
        numj = p.getNumJoints(self.robotId)

        # List the joint names
        print("--------------------------------------------")
        print("JOINT NAMES")
        print("--------------------------------------------")
        for i in range(numj):
            joint = p.getJointInfo(self.robotId, i)
            print("Joint {} name: {}".format(i, joint[1]))

        # Link name dictionary
        _link_name_to_index = {
            p.getBodyInfo(self.robotId)[0].decode('UTF-8'): -1,
        }
        # List the link names
        print("--------------------------------------------")
        print("LINK NAMES")
        print("--------------------------------------------")
        link_names = []
        for _id in range(p.getNumJoints(self.robotId)):
            _name = p.getJointInfo(self.robotId, _id)[12].decode('UTF-8')
            link_names.append(_name)
            _link_name_to_index[_name] = _id
            print("Link name: {} \t index: {}".format(
                _name, _link_name_to_index[_name]))

        # COLLISIONS
        print("--------------------------------------------")
        print("COLLISIONS")
        print("--------------------------------------------")
        # RIGHT LEG GROUP
        # Collide with all joints except those in right leg
        for j in range(6):
            for cj in range(6, numj):
                # Don't collide with right bottom servo on torso
                if self.movingJoints[j] != cj and cj != 4:
                    p.setCollisionFilterPair(self.robotId,
                                             self.robotId,
                                             self.movingJoints[j],
                                             cj,
                                             enableCollision=1)
                    print("COLLISION BETWEEN LINKS: {} AND {}".format(
                        link_names[self.movingJoints[j]], link_names[cj]))

        # LEFT LEG GROUP
        for j in range(6, 12):
            for cj in range(12, numj):
                # Don't collide with left bottom servo on torso
                if self.movingJoints[j] != cj and cj != 12:
                    p.setCollisionFilterPair(self.robotId,
                                             self.robotId,
                                             self.movingJoints[j],
                                             cj,
                                             enableCollision=1)
                    print("COLLISION BETWEEN LINKS: {} AND {}".format(
                        link_names[self.movingJoints[j]], link_names[cj]))
            for cj in range(6):
                # Don't collide with left bottom servo on torso
                if self.movingJoints[j] != cj and cj != 12:
                    p.setCollisionFilterPair(self.robotId,
                                             self.robotId,
                                             self.movingJoints[j],
                                             cj,
                                             enableCollision=1)
                    print("COLLISION BETWEEN LINKS: {} AND {}".format(
                        link_names[self.movingJoints[j]], link_names[cj]))

        # RIGHT ARM GROUP
        for j in range(12, 15):
            for cj in range(15, numj):
                # Don't collide with right top servo on torso
                if self.movingJoints[j] != cj and cj != 2:
                    p.setCollisionFilterPair(self.robotId,
                                             self.robotId,
                                             self.movingJoints[j],
                                             cj,
                                             enableCollision=1)
                    print("COLLISION BETWEEN LINKS: {} AND {}".format(
                        link_names[self.movingJoints[j]], link_names[cj]))
            for cj in range(12):
                # Don't collide with right top servo on torso
                if self.movingJoints[j] != cj and cj != 2:
                    p.setCollisionFilterPair(self.robotId,
                                             self.robotId,
                                             self.movingJoints[j],
                                             cj,
                                             enableCollision=1)
                    print("COLLISION BETWEEN LINKS: {} AND {}".format(
                        link_names[self.movingJoints[j]], link_names[cj]))

        # LEFT ARM GROUP
        for j in range(15, 18):
            for cj in range(18, numj):
                # Don't collide with right top servo on torso
                if self.movingJoints[j] != cj and cj != 3:
                    p.setCollisionFilterPair(self.robotId,
                                             self.robotId,
                                             self.movingJoints[j],
                                             cj,
                                             enableCollision=1)
                    print("COLLISION BETWEEN LINKS: {} AND {}".format(
                        link_names[self.movingJoints[j]], link_names[cj]))
            for cj in range(15):
                # Don't collide with right top servo on torso
                if self.movingJoints[j] != cj and cj != 3:
                    p.setCollisionFilterPair(self.robotId,
                                             self.robotId,
                                             self.movingJoints[j],
                                             cj,
                                             enableCollision=1)
                    print("COLLISION BETWEEN LINKS: {} AND {}".format(
                        link_names[self.movingJoints[j]], link_names[cj]))
        print("--------------------------------------------")
        print("--------------------------------------------")

        # Change Right and Left Foot Dynamics
        roll_fric = 0.01
        lat_fric = 1000.0
        spin_fric = 1.0
        p.changeDynamics(
            self.robotId,
            11,
            # prevents sliding
            lateralFriction=lat_fric,
            # prevents spinning in place
            spinningFriction=spin_fric,
            # slip by rolling (high to encourage earlier
            # falling so robot learns to walk without
            # potentially causing this
            rollingFriction=roll_fric)
        p.changeDynamics(
            self.robotId,
            19,
            # prevernts sliding
            lateralFriction=lat_fric,
            # prevents spinning in place
            spinningFriction=spin_fric,
            # slip by rolling (high to encourage earlier
            # falling so robot learns to walk without
            # potentially causing this
            rollingFriction=roll_fric)

        # Better performance (realism) if damping for all joints turned off
        # These dampings essentially act as aerodynamic drag
        # Default damping is 0.04
        for j in range(p.getNumJoints(self.robotId)):
            p.changeDynamics(self.robotId,
                             j,
                             linearDamping=0.0,
                             angularDamping=0.0,
                             restitution=0.0)

        # for joint in self.movingJoints:
        #     p.changeDynamics(self.robotId, joint, maxJointVelocity=8.76)

    def reset(self):
        p.resetBasePositionAndOrientation(self.robotId,
                                          posObj=self.StartPos,
                                          ornObj=self.StartOrientation)
        for joint in self.movingJoints:
            p.resetJointState(self.robotId, joint, 0)

        # time.sleep(2)
        self.move_joints(np.zeros(18))
        for i in range(2 * self.sim_stepsize):
            p.stepSimulation()

        # time.sleep(0.1)

        observation = self.compute_observation()
        self._publish_reward(self.cumulated_episode_reward, self.episode_num)
        self.episode_num += 1
        self.moving_avg_counter += 1
        self.cumulated_episode_reward = 0
        self.episode_timestep = 0
        return observation

    def _publish_reward(self, reward, episode_number):
        """
        This function publishes the given reward in the reward topic for
        easy access from ROS infrastructure.
        :param reward:
        :param episode_number:
        :return:
        """

        # Now Calculate Moving Avg
        if self.moving_avg_counter >= self.moving_avg_buffer_size:
            self.moving_avg_counter = 0
        self.moving_avg_buffer[
            self.moving_avg_counter] = self.cumulated_episode_reward
        # Only publish moving avg if enough samples
        if self.episode_num >= self.moving_avg_buffer_size:
            moving_avg_reward = np.average(self.moving_avg_buffer)
        else:
            moving_avg_reward = np.nan

        print(
            "Episode #{} \tTotal Timesteps: {} \nReward: {} \tMA Reward: {}\n".
            format(episode_number, self.total_timesteps, reward,
                   moving_avg_reward))

    def step(self, action):
        # Convert agent actions into real actions
        env_action = np.zeros(18)
        # print("MESS {}".format(env_action))

        for i in range(len(action)):
            # Convert action from [-1, 1] to real env values
            env_action[i] = self.agent_to_env(self.env_ranges[i], action[i])

        # print("ENV ACTION {}".format(env_action))
        # p.stepSimulation()
        # self.move_joints(np.ones(18))
        self.move_joints(env_action)
        # p.stepSimulation()
        for i in range(self.sim_stepsize):
            # time.sleep(0.005)
            p.stepSimulation()

        # time.sleep(2)

        observation = self.compute_observation()
        done = self.compute_done()
        reward = self.compute_reward()
        self.cumulated_episode_reward += reward
        self.episode_timestep += 1
        self.total_timesteps += 1
        return observation, reward, done, {}

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

    def move_joints(self, action):
        """
        RIGHT LEG:
        Joint 5 name: rb_servo_r_hip
        Joint 6 name: r_hip_r_thigh
        Joint 7 name: r_thigh_r_knee
        Joint 9 name: r_knee_r_shin
        Joint 10 name: r_shin_r_ankle
        Joint 11 name: r_ankle_r_foot - CONTACT

        LEFT LEG:
        Joint 13 name: lb_servo_l_hip
        Joint 14 name: l_hip_l_thigh
        Joint 15 name: l_thigh_l_knee
        Joint 17 name: l_knee_l_shin
        Joint 18 name: l_shin_l_ankle
        Joint 19 name: l_ankle_l_foot - CONTACT

        RIGHT ARM:
        Joint 20 name: torso_r_shoulder
        Joint 21 name: r_shoulder_rs_servo
        Joint 24 name: re_servo_r_elbow

        LEFT ARM:
        Joint 26 name: torso_l_shoulder
        Joint 27 name: l_shoulder_ls_servo
        Joint 30 name: le_servo_l_elbow

        TOTAL: 18
        """
        p.setJointMotorControlArray(
            bodyUniqueId=self.robotId,
            jointIndices=self.movingJoints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=action,
            # maxVelocities=np.ones(18) * 8.0,
            # targetVelocities=np.zeros(18),
            forces=np.ones(18) * 0.15)

        # for i, key in enumerate(self.movingJoints):
        #     p.setJointMotorControl2(bodyUniqueId=self.robotId,
        #                             jointIndex=key,
        #                             controlMode=p.POSITION_CONTROL,
        #                             targetPosition=action[i])
        # for i in range(len(self.movingJoints)):
        #     p.setJointMotorControl2(bodyUniqueId=self.robotId,
        #                             jointIndex=self.movingJoints[i],
        #                             controlMode=p.POSITION_CONTROL,
        #                             targetPosition=action[i])
        # print("INDEX: {}".format(i))
        # print("KEY: {}".format(key))

    def compute_observation(self):
        baseOri = np.array(p.getBasePositionAndOrientation(self.robotId))
        JointStates = p.getJointStates(self.robotId, self.movingJoints)
        BaseAngVel = p.getBaseVelocity(self.robotId)
        left_contact = p.getContactPoints(self.robotId, self.plane, 19)
        # print(len(left_contact))
        if len(left_contact) > 0:
            self.left_contact = 1
            # print("LEFT CONTACT")
        else:
            # print("LEFT AIR")
            self.left_contact = 0
        right_contact = p.getContactPoints(self.robotId, self.plane, 11)
        if len(right_contact) > 0:
            self.right_contact = 1
            # print("RIGHT CONTACT")
        else:
            # print("RIGHT AIR")
            self.right_contact = 0

        self.torso_z = baseOri[0][2]
        self.torso_y = baseOri[0][1]
        roll, pitch, yaw = p.getEulerFromQuaternion(
            [baseOri[1][0], baseOri[1][1], baseOri[1][2], baseOri[1][3]])
        self.torso_roll = roll
        self.torso_pitch = pitch
        self.torso_yaw = yaw
        self.torso_vx = BaseAngVel[0][0]
        self.torso_w_roll = BaseAngVel[1][0]
        self.torso_w_pitch = BaseAngVel[1][1]
        self.torso_w_yaw = BaseAngVel[1][2]
        self.joint_poses = np.array([
            JointStates[0][0], JointStates[1][0], JointStates[2][0],
            JointStates[3][0], JointStates[4][0], JointStates[5][0],
            JointStates[6][0], JointStates[7][0], JointStates[8][0],
            JointStates[9][0], JointStates[10][0], JointStates[11][0],
            JointStates[12][0], JointStates[13][0], JointStates[14][0],
            JointStates[15][0], JointStates[16][0], JointStates[17][0]
        ])

        observations = np.append(
            self.joint_poses,
            np.array([
                self.torso_z, self.torso_vx, self.torso_roll, self.torso_pitch,
                self.torso_yaw, self.torso_y, self.right_contact,
                self.left_contact
            ]))

        return observations

    def compute_reward(self):
        """
        Return the reward based on the observations given
        """
        reward = 0

        # Reward for being alive
        reward += self.alive_reward
        # Reward for forward velocity
        reward += np.sign(self.torso_vx) * (self.torso_vx * self.vel_weight)**2
        # print("TORSO VX: {}".format(self.torso_vx))
        # Reward for maintaining original height
        """NOTE: TOGGLE BELOW TO ADD ADDITIONAL CONSTRAINTS TO REWARD FCN
        """
        reward -= (np.abs(self.init_height - self.torso_z) *
                   self.height_weight)**2
        # print("HEIGHT PENALTY: {}".format(
        #     (np.abs(self.init_height - self.torso_z) * self.height_weight)**2))
        # Reward for staying on x axis
        reward -= (np.abs(self.torso_y))**2 * self.straight_weight
        # Reward staying upright
        reward -= (np.abs(self.torso_roll))**2 * self.roll_weight
        # Reward for staying upright
        reward -= (np.abs(self.torso_pitch))**2 * self.pitch_weight
        # Reward for facing forward
        reward -= (np.abs(self.torso_yaw))**2 * self.yaw_weight

        # Penalty for having both feet on the ground
        # NOTE: UNSURE ABOUT RESULTS
        if self.right_contact == 1 and self.left_contact == 1:
            reward -= 1

        # Reward for minimal joint actuation
        # NOTE: UNUSED SINCE CANNOT MEASURE ON REAL PLEN
        # for effort in self.joint_efforts:
        #     reward -= effort**2 * self.joint_effort_weight
        # Whether the episode is done due to failure
        if self.dead:
            reward -= self.dead_penalty
            self.dead = False

        # p.addUserDebugLine(lineFromXYZ=(0, 0, 0),
        #                    lineToXYZ=(0.3, 0, 0),
        #                    lineWidth=5,
        #                    lineColorRGB=[0, 255, 0],
        #                    parentObjectUniqueId=self.robotId)
        # p.addUserDebugText("Rewards {}".format(reward), [0, 0, 0.3],
        #                    lifeTime=0.25,
        #                    textSize=2.5,
        #                    parentObjectUniqueId=self.robotId)
        return reward

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
            self.dead = True
        elif self.episode_timestep > self.max_episode_steps and self.torso_x < 1:
            # Terminate episode if plen hasn't moved significantly
            done = True
            self.dead = False
        else:
            done = False
            self.dead = False
        return done

    def close(self):
        print("Ending Simulation")
        p.disconnect()