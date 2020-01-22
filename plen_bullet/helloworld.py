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
print("Number of joints {}".format(numj))
for i in range(10000):
    p.stepSimulation()
    time.sleep(1. / 240.)  # 240hz
p.disconnect()
