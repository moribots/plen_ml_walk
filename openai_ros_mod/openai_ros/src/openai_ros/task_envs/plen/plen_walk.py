import rospy
import numpy as np
from gym import spaces
from openai_ros.robot_envs import plen_env
from gym.envs.registration import register
from geometry_msgs.msg import Point, Twist, Vector3
from tf.transformations import euler_from_quaternion
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
import os


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

        low_act = np.ones(18) * -1
        high_act = low_act * -1
        self.action_space = spaces.Box(low_act, high_act, dtype=np.float32)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, np.inf)

        # Actions and Observations

        self.init_joint_states = np.zeros(18)

        # Desired Velocity
        self.desired_vel = Twist()
        self.desired_vel.linear.x = 1
        # others = 0

        self.desired_yaw = rospy.get_param("/plen/desired_yaw")

        self.accepted_joint_error = rospy.get_param(
            "/plen/accepted_joint_error")
        self.update_rate = rospy.get_param("/plen/update_rate")

        self.dec_obs = rospy.get_param("/plen/number_decimals_precision_obs")

        self.desired_force = rospy.get_param("/plen/desired_force")

        self.max_x_pos = rospy.get_param("/plen/max_x_pos")
        self.max_y_pos = rospy.get_param("/plen/max_y_pos")

        self.min_height = rospy.get_param("/plen/min_height")
        self.max_height = rospy.get_param("/plen/max_height")

        self.distance_from_desired_point_max = rospy.get_param(
            "/plen/distance_from_desired_point_max")

        self.max_incl_roll = rospy.get_param("/plen/max_incl")
        self.max_incl_pitch = rospy.get_param("/plen/max_incl")
        self.max_contact_force = rospy.get_param("/plen/max_contact_force")

        self.maximum_haa_joint = rospy.get_param("/plen/maximum_haa_joint")
        self.maximum_hfe_joint = rospy.get_param("/plen/maximum_hfe_joint")
        self.maximum_kfe_joint = rospy.get_param("/plen/maximum_kfe_joint")
        self.min_kfe_joint = rospy.get_param("/plen/min_kfe_joint")

        # We place the Maximum and minimum values of observations
        self.joint_ranges_array = {
            "maximum_haa": self.maximum_haa_joint,
            "minimum_haa_joint": -self.maximum_haa_joint,
            "maximum_hfe_joint": self.maximum_hfe_joint,
            "minimum_hfe_joint": self.maximum_hfe_joint,
            "maximum_kfe_joint": self.maximum_kfe_joint,
            "min_kfe_joint": self.min_kfe_joint
        }

        high = np.array([
            self.distance_from_desired_point_max, self.max_incl_roll,
            self.max_incl_pitch, 3.14, self.max_contact_force,
            self.maximum_haa_joint, self.maximum_hfe_joint,
            self.maximum_kfe_joint, self.max_x_pos, self.max_y_pos,
            self.max_height
        ])

        low = np.array([
            0.0, -1 * self.max_incl_roll, -1 * self.max_incl_pitch, -1 * 3.14,
            0.0, -1 * self.maximum_haa_joint, -1 * self.maximum_hfe_joint,
            self.min_kfe_joint, -1 * self.max_x_pos, -1 * self.max_y_pos,
            self.min_height
        ])

        self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE===>" + str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>" +
                       str(self.observation_space))

        # Rewards
        self.weight_joint_position = rospy.get_param(
            "/plen/rewards_weight/weight_joint_position")
        self.weight_contact_force = rospy.get_param(
            "/plen/rewards_weight/weight_contact_force")
        self.weight_orientation = rospy.get_param(
            "/plen/rewards_weight/weight_orientation")
        self.weight_distance_from_des_point = rospy.get_param(
            "/plen/rewards_weight/weight_distance_from_des_point")

        self.alive_reward = rospy.get_param("/plen/alive_reward")
        self.done_reward = rospy.get_param("/plen/done_reward")

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(PlenWalkEnv, self).__init__(ros_ws_abspath)

        rospy.logdebug("END PlenWalkEnv INIT...")

    def env_to_agent(self, env_range, env_val):
        # Convert using y = mx + b
        agent_range = [-1, 1]
        m = (agent_range[1] - agent_range[0]) / (env_range[1] - env_range[0])
        b = agent_range[1] - (m * env_range[1])
        agent_val = m * env_val + b
        return agent_val

    def agent_to_env(self, env_range, agent_val):
        # Convert using y = mx + b
        agent_range = [-1, 1]
        m = (env_range[1] - env_range[0]) / (agent_range[1] - agent_range[0])
        b = env_range[1] - (m * agent_range[1])
        env_val = m * agent_val + b
        return env_val

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        # TODO

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
        # TODO: Move robot

    def _get_obs(self):
        """
        Here we define what sensor data of our robots observations
        To know which Variables we have acces to, we need to read the
        MyRobotEnv API DOCS
        :return: observations
        """
        # TODO
        return observations

    def _is_done(self, observations):
        """
        Decide if episode is done based on the observations
        """
        # TODO
        return done

    def _compute_reward(self, observations, done):
        """
        Return the reward based on the observations given
        """
        # TODO
        return reward
        
    # Internal TaskEnv Methods