import pybullet
from pybullet_utils import bullet_client

import gym


class Scene:
    """A base class for single- and multiplayer scenes"""
    def __init__(self, bullet_client=bullet_client.BulletClient(connection_mode=pybullet.GUI), gravity=-9.81, timestep=0.0165, frame_skip=4):
        self._p = bullet_client
        self.np_random, seed = gym.utils.seeding.np_random(None)
        self.timestep = timestep
        self.frame_skip = frame_skip

        self.dt = self.timestep * self.frame_skip
        self.cpp_world = World(self._p, gravity, timestep, frame_skip)

    def global_step(self):
        """
        The idea is: apply motor torques for all robots, then call global_step(), then collect
        observations from robots using step() with the same action.
        """
        self.cpp_world.step(self.frame_skip)


class World:
    def __init__(self, bullet_client, gravity, timestep, frame_skip):
        self._p = bullet_client
        self.gravity = gravity
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.numSolverIterations = 5
        self.clean_everything()

    def clean_everything(self):
        # p.resetSimulation()
        self._p.setGravity(0, 0, self.gravity)
        self._p.setDefaultContactERP(0.9)
        # print("self.numSolverIterations=",self.numSolverIterations)
        self._p.setPhysicsEngineParameter(
            fixedTimeStep=self.timestep * self.frame_skip,
            numSolverIterations=self.numSolverIterations,
            numSubSteps=self.frame_skip)

    def step(self, frame_skip):
        self._p.stepSimulation()