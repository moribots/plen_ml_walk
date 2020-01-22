#!/usr/bin/env python
import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")
StartPos = [0, 0, 0.14]
StartOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF("plen.urdf", StartPos, StartOrientation)
numj = p.getNumJoints(boxId)
Pos, Orn = p.getBasePositionAndOrientation(boxId)
print(Pos, Orn)
# print("Number of joints {}".format(numj))
joint = []
# for i in range(32):
#     joint = p.getJointInfo(boxId, i)
#     print("Joint {} name: {}".format(i, joint[1]))
for i in range(10000):
    # Control Motors
    maxVelocity = 2
    mode = p.POSITION_CONTROL
    p.stepSimulation()
    time.sleep(1. / 240.)  # 240hz
    if i == 1000:
    	p.setJointMotorControl2(boxId, 21, controlMode=mode, maxVelocity=maxVelocity, targetPosition=1)
p.disconnect()

"""
RIGHT LEG:
Joint 5 name: rb_servo_r_hip
Joint 6 name: r_hip_r_thigh
Joint 7 name: r_thigh_r_knee
Joint 9 name: r_knee_r_shin
Joint 10 name: r_shin_r_ankle
Joint 11 name: r_ankle_r_foot

LEFT LEG:
Joint 13 name: lb_servo_l_hip
Joint 14 name: l_hip_l_thigh
Joint 15 name: l_thigh_l_knee
Joint 17 name: l_knee_l_shin
Joint 18 name: l_shin_l_ankle
Joint 19 name: l_ankle_l_foot

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
