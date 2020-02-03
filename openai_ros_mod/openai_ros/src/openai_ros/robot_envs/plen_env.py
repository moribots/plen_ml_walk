import numpy
import rospy
import time
from openai_ros import robot_gazebo_env
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Vector3
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from openai_ros.openai_ros_common import ROSLauncher
from openai_ros.task_envs.plen.joint_publisher import JointPub


class PlenEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all PlenEnv environments.
    """
    def __init__(self, ros_ws_abspath):
        """
        Initializes a new PlenEnv environment.

        Generic ROS Interface for PLEN. Initializes services, publishers,
        subscribers, controllers, sensors. Can unpause to allow data stream
        to flow, or pause to manipulate environment without affecting PLEN.
        Can load/unload controllers in case there is a compatibility issue
        between the controller plugin and tf (sometimes, ROS Time is an issue).

        Sensor Topic List:
        * /plen/odom: Position and orientation in Global space
        * /plen/imu/data: Virtual IMU data (vel, acc)
        * /plen/right_foot_contact: Contact sensor info for right foot
        * /plen/left_foot_contact: Contact sensor info for left foot
        * /plen/joint_states: Joint State information (Feedback, Observations)

        Actuators Topic List:
        * /plen/jc#_pc (# from 1 to 18) for each joint as described below

        # RIGHT LEG:
        # Joint 1 name: rb_servo_r_hip
        # Joint 2 name: r_hip_r_thigh
        # Joint 3 name: r_thigh_r_knee
        # Joint 4 name: r_knee_r_shin
        # Joint 5 name: r_shin_r_ankle
        # Joint 6 name: r_ankle_r_foot

        # LEFT LEG:
        # Joint 7 name: lb_servo_l_hip
        # Joint 8 name: l_hip_l_thigh
        # Joint 9 name: l_thigh_l_knee
        # Joint 10 name: l_knee_l_shin
        # Joint 11 name: l_shin_l_ankle
        # Joint 12 name: l_ankle_l_foot

        # RIGHT ARM:
        # Joint 13 name: torso_r_shoulder
        # Joint 14 name: r_shoulder_rs_servo
        # Joint 15 name: re_servo_r_elbow

        # LEFT ARM:
        # Joint 16 name: torso_l_shoulder
        # Joint 17 name: l_shoulder_ls_servo
        # Joint 18 name: le_servo_l_elbow

        Args:
        """
        rospy.logdebug("Start PlenEnv INIT...")

        # We launch the ROSlaunch that spawns the robot into the world
        ROSLauncher(rospackage_name="plen_ros",
                    launch_file_name="gazebo_plen.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Variables that we give through the constructor of the
        # Parent Class (RobotGazeboEnv).
        """ JOINT ACTUATORS (PUB)
        """
        self.joints = JointPub()
        self._check_all_publishers_ready()

        # Namespace
        self.robot_name_space = "plen"
        self.controllers_list = [
            'joint_state_controller', '/plen/j1_pc', '/plen/j2_pc',
            '/plen/j3_pc', '/plen/j4_pc', '/plen/j5_pc', '/plen/j6_pc',
            '/plen/j7_pc', '/plen/j8_pc', '/plen/j9_pc', '/plen/j10_pc',
            '/plen/j11_pc', '/plen/j12_pc', '/plen/j13_pc', '/plen/j14_pc',
            '/plen/j15_pc', '/plen/j16_pc', '/plen/j17_pc', '/plen/j18_pc'
        ]

        # We launch the init function of the
        # Parent Class robot_gazebo_env.RobotGazeboEnv
        # INTERFACE WITH PARENT CLASS USING SUPER
        # CHILD METHODS TAKE PRECEDENCE WITH DUPLICATES
        super(PlenEnv,
              self).__init__(controllers_list=self.self.controllers_list,
                             robot_name_space=self.robot_name_space,
                             reset_controls=False,
                             start_init_physics_parameters=False,
                             reset_world_or_sim="WORLD")

        rospy.logdebug("PlenEnv unpause...")
        self.gazebo.unpauseSim()
        # self.controllers_object.reset_controllers()

        self._check_all_systems_ready()
        """ REPLACE BELOW WITH SENSORS
        """
        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber("/odom", Odometry, self._odom_callback)
        # We use the IMU for orientation and linearacceleration detection
        rospy.Subscriber("/plen/imu/data", Imu, self._imu_callback)
        # We use it to get the contact force, to know if its in the air or stumping too hard.
        rospy.Subscriber("/plen/right_foot_contact", ContactsState,
                         self._right_contact_callback)
        rospy.Subscriber("/plen/left_foot_contact", ContactsState,
                         self._left_contact_callback)
        # We use it to get the joints positions and calculate the reward associated to it
        rospy.Subscriber("/plen/joint_states", JointState,
                         self._joints_state_callback)

        self.gazebo.pauseSim()

        rospy.logdebug("Finished PlenEnv INIT...")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        rospy.logdebug("PlenEnv check_all_systems_ready...")
        self._check_all_sensors_ready()
        rospy.logdebug("END PlenEnv _check_all_systems_ready...")
        return True

    """ CHECK ALL SUBSCRIBERS AND PUBLISHERS READY
    """
    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL SENSORS READY")
        self._check_odom_ready()
        self._check_imu_ready()
        self._check_rightfoot_contactsensor_state_ready()
        self._check_joint_states_ready()
        rospy.logdebug("ALL SENSORS READY")

    def _check_odom_ready(self):
        self.odom = None
        rospy.logdebug("Waiting for /odom to be READY...")
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message("/odom",
                                                   Odometry,
                                                   timeout=1.0)
                rospy.logdebug("Current /odom READY=>")

            except:
                rospy.logerr(
                    "Current /odom not ready yet, retrying for getting odom")
        return self.odom

    def _check_imu_ready(self):
        self.imu = None
        rospy.logdebug("Waiting for /plen/imu/data to be READY...")
        while self.imu is None and not rospy.is_shutdown():
            try:
                self.imu = rospy.wait_for_message("/plen/imu/data",
                                                  Imu,
                                                  timeout=1.0)
                rospy.logdebug("Current /plen/imu/data READY=>")

            except:
                rospy.logerr(
                    "Current /plen/imu/data not ready yet, retrying for getting imu"
                )
        return self.imu

    def _check_rightfoot_contactsensor_state_ready(self):
        self.rightfoot_contactsensor_state = None
        rospy.logdebug("Waiting for /plen/right_foot_contact to be READY...")
        while self.rightfoot_contactsensor_state is None and not rospy.is_shutdown(
        ):
            try:
                self.rightfoot_contactsensor_state = rospy.wait_for_message(
                    "/plen/right_foot_contact", ContactsState, timeout=1.0)
                rospy.logdebug("Current /plen/right_foot_contact READY=>")

            except:
                rospy.logerr(
                    "Current /plen/right_foot_contact not ready yet, retrying for getting /plen/right_foot_contact"
                )
        return self.rightfoot_contactsensor_state

    def _check_leftfoot_contactsensor_state_ready(self):
        self.leftfoot_contactsensor_state = None
        rospy.logdebug("Waiting for /plen/left_foot_contact to be READY...")
        while self.leftfoot_contactsensor_state is None and not rospy.is_shutdown(
        ):
            try:
                self.leftfoot_contactsensor_state = rospy.wait_for_message(
                    "/plen/left_foot_contact", ContactsState, timeout=1.0)
                rospy.logdebug("Current /plen/left_foot_contact READY=>")

            except:
                rospy.logerr(
                    "Current /plen/left_foot_contact not ready yet, retrying for getting /plen/left_foot_contact"
                )
        return self.rightfoot_contactsensor_state

    def _check_joint_states_ready(self):
        self.joint_states = None
        rospy.logdebug("Waiting for /plen/joint_states to be READY...")
        while self.joint_states is None and not rospy.is_shutdown():
            try:
                self.joint_states = rospy.wait_for_message(
                    "/plen/joint_states", JointState, timeout=1.0)
                rospy.logdebug("Current /plen/joint_states READY=>")

            except:
                rospy.logerr(
                    "Current /plen/joint_states not ready yet, retrying for getting joint_states"
                )
        return self.joint_states

    def _odom_callback(self, data):
        self.odom = data

    def _imu_callback(self, data):
        self.imu = data

    def _right_contact_callback(self, data):
        self.rightfoot_contactsensor_state = data

    def _left_contact_callback(self, data):
        self.leftfoot_contactsensor_state = data

    def _joints_state_callback(self, data):
        self.joint_states = data

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.joints.set_init_pose

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode. Gravity?
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
           Done in PLEN Task Env.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        # Given the action selected by the learning algorithm,
        # we perform the corresponding movement of the robot

        # 1st, decide which action corresponsd to which joint is incremented
        next_action_position = self.get_action_to_position(action)

        # We move it to that pos
        self.gazebo.unpauseSim()
        self.plen_joint_pubisher_object.move_joints(next_action_position)
        # Then we send the command to the robot and let it go
        # for running_step seconds
        time.sleep(self.running_step)
        self.gazebo.pauseSim()

        # We now process the latest data saved in the class state to calculate
        # the state and the rewards. This way we guarantee that they work
        # with the same exact data.
        # Generate State based on observations
        observation = self.plen_state_object.get_observations()

        # finally we get an evaluation based on what happened in the sim
        reward, done = self.plen_state_object.process_data()

        # Get the State Discrete Stringuified version of the observations
        state = self.get_state(observation)

        return state, reward, done

    def _get_obs(self):
        """Returns the observation.
           Done in Plen Task Env.
        """
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
           Done in Plen Task Env.
        """
        raise NotImplementedError()

    # Methods that the TrainingEnvironment will need.
    # ----------------------------

    def check_array_similar(self, ref_value_array, check_value_array, epsilon):
        """
        It checks if the check_value id similar to the ref_value
        """
        rospy.logdebug("ref_value_array=" + str(ref_value_array))
        rospy.logdebug("check_value_array=" + str(check_value_array))
        return numpy.allclose(ref_value_array, check_value_array, atol=epsilon)

    def get_odom(self):
        return self.odom

    def get_imu(self):
        return self.imu

    def get_rightfoot_contactsensor_state(self):
        return self.rightfoot_contactsensor_state

    def get_leftfoot_contactsensor_state(self):
        return self.leftfoot_contactsensor_state

    def get_joint_states(self):
        return self.joint_states


if __name__ == "__main__":
    rospy.logerr("THIS SCRIPT SHOULD NOT BE LAUNCHED AS MAIN")
