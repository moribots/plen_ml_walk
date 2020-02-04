import rospy
import numpy as np
from gym import spaces
from openai_ros.robot_envs import plen_env
from gym.envs.registration import register
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import JointState
from tf.transformations import euler_from_quaternion
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
import os
import time

register(
    id='PlenWalkEnv-v0',
    entry_point='plen_walk:PlenWalkEnv',
    timestep_limit=1000,  # Time Step Limit Per Episode
)


class PlenWalkEnv(plen_env.PlenEnv):
    def __init__(self):
        """
        Make PLEN learn how to Walk
        """
        rospy.logdebug("Start PlenWalkEnv INIT...")

        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/plenros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        ROSLauncher(rospackage_name="legged_robots_sims",
                    launch_file_name="start_world.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file=
                               "src/openai_ros/task_envs/plen/config",
                               yaml_file_name="plen_walk.yaml")

        # How long to step the simulation for (sec)
        self.running_step = 0.05

        # Agent Action Space
        low_act = np.ones(18) * -1
        high_act = low_act * -1
        self.action_space = spaces.Box(low_act, high_act, dtype=np.float32)

        # Environment Action Space
        self.env_ranges = [
            [-1.7, 1.7],  # RIGHT LEG
            [-1.54, 0.12],
            [-1.7, 0.75],
            [-0.2, 0.95],
            [-0.95, 1.54],
            [-0.45, 0.8],
            [-1.7, 1.7],  # LEFT LEG
            [-0.12, 1.54],
            [-0.75, 1.7],
            [-0.95, 0.2],
            [-1.54, 0.95],
            [-0.8, 0.45],
            [-1.7, 1.7],  # RIGHT ARM
            [-0.15, 1.7],
            [-0.2, 0.5],
            [-1.7, 1.7],  # LEFT ARM
            [-0.15, 1.7],
            [-0.2, 0.5]
        ]

        # Possible Rewards
        self.reward_range = (-np.inf, np.inf)

        # Observation Values

        # TODO: REPLACE SCALARS BY .YAML GET PARAM

        # JOINTS (see self.env_ranges)
        # Low values of Joint Position Space
        self.joints_low = []
        # High values of Joint Position Space
        self.joints_high = []
        for j_state in self.env_ranges:
            self.joints_low.append(j_state[0])
            self.joints_high.append(j_state[1])

        # JOINT EFFORT
        self.joint_effort_low = [-0.15] * 18
        self.joint_effort_high = [0.15] * 18

        # TORSO HEIGHT (0, 0.25)
        self.torso_height_min = 0
        self.torso_height_max = 0.25

        # TORSO TWIST (x) (-inf, inf)
        self.torso_vx_min = -np.inf
        self.torso_vx_max = np.inf

        # TORSO ROLL (-pi, pi)
        self.torso_roll_min = -np.pi
        self.torso_roll_max = np.pi

        # TORSO PITCH (-pi, pi)
        self.torso_pitch_min = -np.pi
        self.torso_pitch_max = np.pi

        # TORSO DEVIATION FROM X AXIS (-inf, inf)
        self.torso_y_min = -np.inf
        self.torso_y_max = np.inf

        # RIGHT FOOT CONTACT (0, 1)
        self.rfs_min = 0
        self.rfs_max = 1

        # LEFT FOOT CONTACT (0, 1)
        self.lfs_min = 0
        self.lfs_max = 1

        obs_low = np.array([
            self.joints_low, self.joint_effort_low, self.torso_height_min,
            self.torso_vx_min, self.torso_roll_min, self.torso_pitch_min,
            self.torso_y_min, self.rfs_min, self.lfs_min
        ])

        obs_high = np.array([
            self.joints_high, self.joint_effort_high, self.torso_height_max,
            self.torso_vx_max, self.torso_roll_max, self.torso_pitch_max,
            self.torso_y_max, self.rfs_max, self.lfs_max
        ])

        self.observation_space = spaces.Box(obs_low, obs_high)

        rospy.logdebug("ACTION SPACES TYPE===>" + str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>" +
                       str(self.observation_space))

        # OBSERVATION SUBSCRIBERS

        # Odometry (Pose and Twist)
        self.torso = Odometry()
        self.odom_subscriber = rospy.Subscriber('/plen/odom', Odometry,
                                                self.odom_subscriber_callback)

        # Joint Positions and Effort
        # Init
        self.joint_poses = np.zeros(18)
        self.joint_efforts = np.zeros(18)
        # Sub
        self.joint_state_subscriber = rospy.Subscriber(
            '/plen/joint_states', JointState,
            self.joint_state_subscriber_callback)

        # Right Foot Contact
        # Init
        self.right_contact = 1
        # Sub
        self.right_contact_subscriber = rospy.Subscriber(
            '/plen/right_foot_contact', ContactsState,
            self.right_contact_subscriber_callback)

        # Left Foot Contact
        self.left_contact = 1
        # Sub
        self.left_contact_subscriber = rospy.Subscriber(
            '/plen/left_foot_contact', ContactsState,
            self.left_contact_subscriber_callback)

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(PlenWalkEnv, self).__init__(ros_ws_abspath)

        rospy.logdebug("END PlenWalkEnv INIT...")

    def odom_subscriber_callback(self, msg):
        """
            Returns cartesian position and orientation of torso middle
        """
        # TODO
        return 0

    def joint_state_subscriber_callback(self, msg):
        """
            Returns joint positions and efforts
        """
        # TODO
        return 0

    def right_contact_subscriber_callback(self, msg):
        """
            Returns whether right foot has made contact
        """
        # TODO
        return 0

    def left_contact_subscriber_callback(self, msg):
        """
            Returns whether left foot has made contact
        """
        # TODO
        return 0

    def env_to_agent(self, env_range, env_val):
        """ Convert an action from the Environment space
            to the Agent Space ([-1, 1])
        """
        # Convert using y = mx + b
        agent_range = [-1, 1]
        m = (agent_range[1] - agent_range[0]) / (env_range[1] - env_range[0])
        b = agent_range[1] - (m * env_range[1])
        agent_val = m * env_val + b
        return agent_val

    def agent_to_env(self, env_range, agent_val):
        """ Convert an action from the Agent space ([-1, 1])
            to the Environment Space
        """
        # Convert using y = mx + b
        agent_range = [-1, 1]
        m = (env_range[1] - env_range[0]) / (agent_range[1] - agent_range[0])
        b = env_range[1] - (m * agent_range[1])
        env_val = m * agent_val + b
        return env_val

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.joints.set_init_pose

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # TODO

    def _set_action(self, action):
        """
        Move the robot based on the action variable given
        """
        env_action = np.empty(18)
        for i in range(len(action)):
            # Convert action from [-1, 1] to real env values
            env_action[i] = self.agent_to_env(self.env_ranges[i], action[i])

        rospy.logdebug("Executing Action ==>" + str(env_action))

        # Unpause
        self.gazebo.unpauseSim()
        # Move Joints
        self.joints.move_joints(env_action)
        # Let run for running_step seconds
        # time.sleep(self.running_step)
        # Pause
        self.gazebo.pauseSim()

        rospy.logdebug("Action Completed")

    def _get_obs(self):
        """
        Here we define what sensor data of our robots observations

            - Twist
            - Torso Height
            - Torso Pitc
            - Torso Roll
            - Torso y position
            - Joint Positions
            - Joint efforts
            - Right foot contact
            - Left foot contact
        """
        # TODO
        return observations

    def _is_done(self, observations):
        """
        Decide if episode is done based on the observations

            - Pitch is above or below pi/2
            - Roll is above or below pi/2
            - Height is below height thresh
            - y position (abs) is above y thresh
            - episode timesteps above limit
        """
        # TODO
        return done

    def _compute_reward(self, observations, done):
        """
        Return the reward based on the observations given
        """
        # TODO
        return reward