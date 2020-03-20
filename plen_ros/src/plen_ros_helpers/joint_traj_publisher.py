#!/usr/bin/env python

import rospy
import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from gazebo_msgs.srv import JointRequest


class JointTrajPub(object):
    def __init__(self, joint_name_list, joint_name_string):
        # rospy.loginfo('Waiting for joint trajectory Publisher')
        self.jtp = rospy.Publisher('/plen/joint_trajectory_controller/command',
                                   JointTrajectory,
                                   queue_size=1)
        self.clear_forces = rospy.ServiceProxy("/gazebo/clear_joint_forces",
                                               JointRequest)
        self.joint_name_list = joint_name_list
        self.jtp_zeros = np.zeros(len(joint_name_list))
        self.joint_name_string = joint_name_string

    def move_joints(self, pos):
        self.check_joints_connection()
        rospy.wait_for_service("/gazebo/clear_joint_forces")
        for name in self.joint_name_list:
            self.clear_forces(joint_name=name)
        # self.clear_forces(joint_name=self.joint_name_string)
        jtp_msg = JointTrajectory()
        jtp_msg.joint_names = self.joint_name_list
        point = JointTrajectoryPoint()
        point.positions = pos
        # See urdf for vel (FEETECH FS-90 Servo)
        point.velocities = self.jtp_zeros  # np.ones(18) * 8.0
        # No Acc Data
        point.accelerations = self.jtp_zeros
        # See urdf for effort (FEETECH FS-90 Servo)
        point.effort = np.ones(18) * 0.15
        # one ns, the minimum time
        point.time_from_start = rospy.Duration(1e-9)
        jtp_msg.points.append(point)
        self.jtp.publish(jtp_msg)
        # rospy.sleep(0.05)

    def set_init_pose(self, pos):
        self.check_joints_connection()
        rospy.wait_for_service("/gazebo/clear_joint_forces")
        for name in self.joint_name_list:
            self.clear_forces(joint_name=name)
        jtp_msg = JointTrajectory()
        self.jtp.publish(jtp_msg)
        jtp_msg = JointTrajectory()
        jtp_msg.joint_names = self.joint_name_list
        point = JointTrajectoryPoint()
        point.positions = pos
        # See urdf for vel (FEETECH FS-90 Servo)
        point.velocities = self.jtp_zeros  # np.ones(18) * 8.0
        # No Acc Data
        point.accelerations = self.jtp_zeros  # np.ones(18) * 8.0
        # See urdf for effort (FEETECH FS-90 Servo)
        point.effort = np.ones(18) * 0.15
        # one ns, the minimum time
        point.time_from_start = rospy.Duration(1e-9)
        jtp_msg.points.append(point)
        self.jtp.publish(jtp_msg)
        # rospy.sleep(0.05)

    def check_joints_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while (self.jtp.get_num_connections() == 0):
            rospy.logdebug("No susbribers to joint " +
                           "yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # Avoid error when world is rested, time when backwards.
                pass
        # rospy.logdebug("All Publishers READY")