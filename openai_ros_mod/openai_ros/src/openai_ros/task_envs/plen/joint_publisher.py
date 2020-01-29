#!/usr/bin/env python

import rospy
import numpy as np
import time
import math
from std_msgs.msg import String
from std_msgs.msg import Float64


class JointPub(object):
    def __init__(self):

        # RIGHT LEG
        self.rhip = rospy.Publisher('/plen/j1_pc/command',
                                    Float64,
                                    queue_size=1)
        self.rthigh = rospy.Publisher('/plen/j2_pc/command',
                                      Float64,
                                      queue_size=1)
        self.rknee = rospy.Publisher('/plen/j3_pc/command',
                                     Float64,
                                     queue_size=1)
        self.rshin = rospy.Publisher('/plen/j4_pc/command',
                                     Float64,
                                     queue_size=1)
        self.rankle = rospy.Publisher('/plen/j5_pc/command',
                                      Float64,
                                      queue_size=1)
        self.rfoot = rospy.Publisher('/plen/j6_pc/command',
                                     Float64,
                                     queue_size=1)
        # LEFT LEG
        self.lhip = rospy.Publisher('/plen/j7_pc/command',
                                    Float64,
                                    queue_size=1)
        self.lthigh = rospy.Publisher('/plen/8_pc/command',
                                      Float64,
                                      queue_size=1)
        self.lknee = rospy.Publisher('/plen/j9_pc/command',
                                     Float64,
                                     queue_size=1)
        self.lshin = rospy.Publisher('/plen/j10_pc/command',
                                     Float64,
                                     queue_size=1)
        self.lankle = rospy.Publisher('/plen/j11_pc/command',
                                      Float64,
                                      queue_size=1)
        self.lfoot = rospy.Publisher('/plen/j12_pc/command',
                                     Float64,
                                     queue_size=1)
        # RIGHT ARM
        self.rshoulder = rospy.Publisher('/plen/j13_pc/command',
                                         Float64,
                                         queue_size=1)
        self.rarm = rospy.Publisher('/plen/j14_pc/command',
                                    Float64,
                                    queue_size=1)
        self.relbow = rospy.Publisher('/plen/j15_pc/command',
                                      Float64,
                                      queue_size=1)
        # LEFT ARM
        self.lshoulder = rospy.Publisher('/plen/j16_pc/command',
                                         Float64,
                                         queue_size=1)
        self.larm = rospy.Publisher('/plen/j17_pc/command',
                                    Float64,
                                    queue_size=1)
        self.lelbow = rospy.Publisher('/plen/j18_pc/command',
                                      Float64,
                                      queue_size=1)

        self.publishers_array = [
            self.rhip, self.rthigh, self.rknee, self.rshin, self.rankle,
            self.rfoot, self.lhip, self.lthigh, self.lknee, self.lshin,
            self.lankle, self.lfoot, self.rshoulder, self.rarm, self.relbow,
            self.lshoulder, self.larm, self.lelbow
        ]
        # Initial joint state
        self.init_pos = np.zeros(18)

    def set_init_pose(self):
        """
        Sets joints to initial position [0,0,0,...]
        :return:
        """
        self.check_joints_connection()
        self.move_joints(self.init_pos)

    def check_joints_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        i = 0
        for publisher_object in self.publishers_array:
            while (publisher_object.get_num_connections() == 0):
                i += 1
                rospy.logdebug("No susbribers to joint " + str(i) +
                               "yet so we wait and try again")
                try:
                    rate.sleep()
                except rospy.ROSInterruptException:
                    # Avoid error when world is rested, time when backwards.
                    pass
            rospy.logdebug("Joint " + str(i) + "Publisher Connected")
        rospy.logdebug("All Publishers READY")

    def joint_mono_des_callback(self, msg):
        rospy.logdebug(str(msg.joint_state.position))

        self.move_joints(msg.joint_state.position)

    def move_joints(self,
                    joints_array,
                    epsilon=0.05,
                    update_rate=10,
                    time_sleep=0.05,
                    check_position=True):
        """
        It will move the Plen Joints to the given Joint_Array values
        """
        i = 0
        for publisher_object in self.publishers_array:
            joint_value = Float64()
            joint_value.data = joints_array[i]
            rospy.logdebug("JointsPos>>" + str(joint_value))
            publisher_object.publish(joint_value)
            i += 1

        if check_position:
            self.wait_time_for_execute_movement(joints_array, epsilon,
                                                update_rate)
        else:
            self.wait_time_movement_hard(time_sleep=time_sleep)

    def wait_time_for_execute_movement(self, joints_array, epsilon,
                                       update_rate):
        """
        We wait until Joints are where we asked them to be based on the joints_states
        :param joints_array:Joints Values in radians of each of the three joints of Plen leg.
        :param epsilon: Error acceptable in odometry readings.
        :param update_rate: Rate at which we check the joint_states.
        :return:
        """
        rospy.logdebug("START wait_until_twist_achieved...")

        rate = rospy.Rate(update_rate)
        start_wait_time = rospy.get_rostime().to_sec()
        end_wait_time = 0.0

        rospy.logdebug("Desired JointsState>>" + str(joints_array))
        rospy.logdebug("epsilon>>" + str(epsilon))

        while not rospy.is_shutdown():
            current_joint_states = self._check_joint_states_ready()

            values_to_check = [
                current_joint_states.position[0],
                current_joint_states.position[1],
                current_joint_states.position[2]
            ]

            vel_values_are_close = self.check_array_similar(
                joints_array, values_to_check, epsilon)

            if vel_values_are_close:
                rospy.logdebug("Reached JointStates!")
                end_wait_time = rospy.get_rostime().to_sec()
                break
            rospy.logdebug("Not there yet, keep waiting...")
            rate.sleep()
        delta_time = end_wait_time - start_wait_time
        rospy.logdebug("[Wait Time=" + str(delta_time) + "]")

        rospy.logdebug("END wait_until_jointstate_achieved...")

        return delta_time

    def wait_time_movement_hard(self, time_sleep):
        """
        Hard Wait to avoid inconsistencies in times executing actions
        """
        rospy.logdebug("Test Wait=" + str(time_sleep))
        time.sleep(time_sleep)

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


if __name__ == "__main__":
    rospy.logerr("THIS SCRIPT SHOULD NOT BE LAUNCHED AS MAIN")