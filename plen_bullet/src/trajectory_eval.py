#!/usr/bin/env python

import numpy as np

from plen_ros_helpers.td3 import ReplayBuffer, TD3Agent

from plen_bullet import plen_env

from plen_bullet.trajectory_generator import TrajectoryGenerator

import gym
import torch
import os

import time


def main():
    """ The main() function. """

    # GYM Env
    env_name = "PlenWalkEnv-v1"
    seed = 0
    max_timesteps = 4e6

    # Find abs path to this file
    my_path = os.path.abspath(os.path.dirname(__file__))
    results_path = os.path.join(my_path, "../results")
    models_path = os.path.join(my_path, "../models")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    env = gym.make(env_name, render=True, joint_act=True)

    # Set seeds
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print("RECORDED MAX ACTION: {}".format(max_action))

    policy = TD3Agent(state_dim, action_dim, max_action)
    # Optionally load existing policy, replace 9999 with num
    policy_num = 3229999  # 629999 current best policy
    if os.path.exists(models_path + "/" + "plen_walk_gazebo_" +
                      str(policy_num) + "_critic"):
        print("Loading Existing Policy")
        policy.load(models_path + "/" + "plen_walk_gazebo_" + str(policy_num))

    replay_buffer = ReplayBuffer()
    # Optionally load existing policy, replace 9999 with num
    buffer_number = 0  # BY DEFAULT WILL LOAD NOTHING, CHANGE THIS
    if os.path.exists(replay_buffer.buffer_path + "/" + "replay_buffer_" +
                      str(buffer_number) + '.data'):
        print("Loading Replay Buffer " + str(buffer_number))
        replay_buffer.load(buffer_number)
        print(replay_buffer.storage)

    # Evaluate untrained policy and init list for storage
    evaluations = []

    state = env.reset()
    done = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    print("STARTED PLEN_TD3 RL SCRIPT")

    # # RIGHT LEG
    # rhip_traj = []
    # rthigh_traj = []
    # rknee_traj = []
    # rshin_traj = []
    # rankle_traj = []
    # rfoot_traj = []
    # # LEFT LEG
    # lhip_traj = []
    # lthigh_traj = []
    # lknee_traj = []
    # lshin_traj = []
    # lankle_traj = []
    # lfoot_traj = []
    # # RIGHT ARM
    # rshoulder_traj = []
    # rarm_traj = []
    # relbow_traj = []
    # # LEFT ARM
    # lshoulder_traj = []
    # larm_traj = []
    # lelbow_traj = []

    # joint_trajectories = [
    #     rhip_traj, rthigh_traj, rknee_traj, rshin_traj, rankle_traj,
    #     rfoot_traj, lhip_traj, lthigh_traj, lknee_traj, lshin_traj,
    #     lankle_traj, lfoot_traj, rshoulder_traj, rarm_traj, relbow_traj,
    #     lshoulder_traj, larm_traj, lelbow_traj
    # ]

    # Next, load the joint commands for each joint across each timestep

    # RIGHT LEG
    # rhip_cmd = []
    # rthigh_cmd = []
    # rknee_cmd = []
    # rshin_cmd = []
    # rankle_cmd = []
    # rfoot_cmd = []
    # # LEFT LEG
    # lhip_cmd = []
    # lthigh_cmd = []
    # lknee_cmd = []
    # lshin_cmd = []
    # lankle_cmd = []
    # lfoot_cmd = []
    # # RIGHT ARM
    # rshoulder_cmd = []
    # rarm_cmd = []
    # relbow_cmd = []
    # # LEFT ARM
    # lshoulder_cmd = []
    # larm_cmd = []
    # lelbow_cmd = []

    # joint_cmds = [
    #     rhip_cmd, rthigh_cmd, rknee_cmd, rshin_cmd, rankle_cmd, rfoot_cmd,
    #     lhip_cmd, lthigh_cmd, lknee_cmd, lshin_cmd, lankle_cmd, lfoot_cmd,
    #     rshoulder_cmd, rarm_cmd, relbow_traj, lshoulder_cmd, larm_cmd,
    #     lelbow_cmd
    # ]

    joint_names = [
        'rb_servo_r_hip', 'r_hip_r_thigh', 'r_thigh_r_knee', 'r_knee_r_shin',
        'r_shin_r_ankle', 'r_ankle_r_foot', 'lb_servo_l_hip', 'l_hip_l_thigh',
        'l_thigh_l_knee', 'l_knee_l_shin', 'l_shin_l_ankle', 'l_ankle_l_foot',
        'torso_r_shoulder', 'r_shoulder_rs_servo', 're_servo_r_elbow',
        'torso_l_shoulder', 'l_shoulder_ls_servo', 'le_servo_l_elbow'
    ]

    # for i in range(len(joint_names)):
    #     joint_trajectories[i] = np.load("../trajectories/" + joint_names[i] +
    #                                     "_traj.npy")
    #     joint_cmds[i] = np.load("../trajectories/" + joint_names[i] +
    #                             "_cmd.npy")

    traj = TrajectoryGenerator()
    traj.main()

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
    # # RIGHT ARM
    rshoulder_traj = []
    rarm_traj = []
    relbow_traj = []
    # LEFT ARM
    lshoulder_traj = []
    larm_traj = []
    lelbow_traj = []

    # Populate Leg Joints
    # START LEFT
    # for i in range(traj.num_DoubleSupport + traj.num_SingleSupport):
    #     # RIGHT LEG
    #     rhip_traj.append(-traj.foot_start_lfwd[i][0])
    #     rthigh_traj.append(-traj.foot_start_lfwd[i][1])
    #     rknee_traj.append(-traj.foot_start_lfwd[i][2])
    #     rshin_traj.append(-traj.foot_start_lfwd[i][3])
    #     rankle_traj.append(traj.foot_start_lfwd[i][4])
    #     rfoot_traj.append(traj.foot_start_lfwd[i][5])
    #     # LEFT LEG
    #     lhip_traj.append(traj.foot_start_lfwd[i][6])
    #     lthigh_traj.append(traj.foot_start_lfwd[i][7])
    #     lknee_traj.append(traj.foot_start_lfwd[i][8])
    #     lshin_traj.append(traj.foot_start_lfwd[i][9])
    #     lankle_traj.append(-traj.foot_start_lfwd[i][10])
    #     lfoot_traj.append(traj.foot_start_lfwd[i][11])
    #     # if i > traj.num_DoubleSupport:
    #     # RIGHT ARM
    #     rshoulder_traj.append(np.pi / 5)
    #     rarm_traj.append(np.pi / 8)
    #     relbow_traj.append(0)
    #     # LEFT ARM
    #     lshoulder_traj.append(np.pi / 5)
    #     larm_traj.append(np.pi / 8)
    #     lelbow_traj.append(0)

    for p in range(5):
        for i in range(np.size(traj.foot_walk_rfwd, 0)):
            # WALK RIGHT
            # RIGHT LEG
            rhip_traj.append(-traj.foot_walk_rfwd[i][0])
            rthigh_traj.append(-traj.foot_walk_rfwd[i][1])
            rknee_traj.append(-traj.foot_walk_rfwd[i][2])
            rshin_traj.append(-traj.foot_walk_rfwd[i][3])
            rankle_traj.append(traj.foot_walk_rfwd[i][4])
            rfoot_traj.append(traj.foot_walk_rfwd[i][5])
            # LEFT LEG
            lhip_traj.append(traj.foot_walk_rfwd[i][6])
            lthigh_traj.append(traj.foot_walk_rfwd[i][7])
            lknee_traj.append(traj.foot_walk_rfwd[i][8])
            lshin_traj.append(traj.foot_walk_rfwd[i][9])
            lankle_traj.append(-traj.foot_walk_rfwd[i][10])
            lfoot_traj.append(traj.foot_walk_rfwd[i][11])
            # if i > np.size(traj.foot_walk_rfwd, 0) / 2.0:
            # RIGHT ARM
            rshoulder_traj.append(np.pi / 5)
            rarm_traj.append(np.pi / 8)
            relbow_traj.append(0)
            # LEFT ARM
            lshoulder_traj.append(-np.pi / 5)
            larm_traj.append(np.pi / 8)
            lelbow_traj.append(0)

        for i in range(np.size(traj.foot_walk_lfwd, 0)):
            # WALK LEFT
            # RIGHT LEG
            rhip_traj.append(-traj.foot_walk_lfwd[i][0])
            rthigh_traj.append(-traj.foot_walk_lfwd[i][1])
            rknee_traj.append(-traj.foot_walk_lfwd[i][2])
            rshin_traj.append(-traj.foot_walk_lfwd[i][3])
            rankle_traj.append(traj.foot_walk_lfwd[i][4])
            rfoot_traj.append(traj.foot_walk_lfwd[i][5])
            # LEFT LEG
            lhip_traj.append(traj.foot_walk_lfwd[i][6])
            lthigh_traj.append(traj.foot_walk_lfwd[i][7])
            lknee_traj.append(traj.foot_walk_lfwd[i][8])
            lshin_traj.append(traj.foot_walk_lfwd[i][9])
            lankle_traj.append(-traj.foot_walk_lfwd[i][10])
            lfoot_traj.append(traj.foot_walk_lfwd[i][11])
            # if i > np.size(traj.foot_walk_rfwd, 0) / 2.0:
            # RIGHT ARM
            rshoulder_traj.append(np.pi / 5)
            rarm_traj.append(np.pi / 8)
            relbow_traj.append(0)
            # LEFT ARM
            lshoulder_traj.append(-np.pi / 5)
            larm_traj.append(np.pi / 8)
            lelbow_traj.append(0)

    # RIGHT ARM
    # rshoulder_traj = [0] * len(rhip_traj)
    # rarm_traj = [0] * len(rhip_traj)
    # relbow_traj = [0] * len(rhip_traj)
    # # LEFT ARM
    # lshoulder_traj = [0] * len(rhip_traj)
    # larm_traj = [0] * len(rhip_traj)
    # lelbow_traj = [0] * len(rhip_traj)

    # Correct For neg angles

    joint_trajectories = [
        rhip_traj, rthigh_traj, rknee_traj, rshin_traj, rankle_traj,
        rfoot_traj, lhip_traj, lthigh_traj, lknee_traj, lshin_traj,
        lankle_traj, lfoot_traj, rshoulder_traj, rarm_traj, relbow_traj,
        lshoulder_traj, larm_traj, lelbow_traj
    ]

    bend_legs = traj.bend[:][0]
    for i in range(6):
        bend_legs = np.append(bend_legs, 0)

    bend_legs[13] = 0.5
    bend_legs[16] = 0.5

    for i in range(4):
        bend_legs[i] = -bend_legs[i]

    bend_legs[10] = -bend_legs[10]

    # Save Trajectories
    results_path = os.path.join(my_path, "../trajectories/")

    # SAVE JOINT TRAJECTORIES
    for i in range(len(joint_trajectories)):
        np.save(results_path + joint_names[i] + "_traj", joint_trajectories[i])

    # Save Leg Bend Traj
    np.save(results_path + "bend_traj", bend_legs)

    print("BEND")
    for i in range(20):
        # Perform action
        time.sleep(1. / 20.)
        next_state, reward, done, _ = env.step(bend_legs)

    print("WALK")
    for t in range(len(joint_trajectories[0])):

        time.sleep(1. / 20.)

        episode_timesteps += 1
        # Deterministic Policy Action
        action = np.zeros(18)
        # print("t: {}".format(t))
        for j in range(len(joint_trajectories)):
            # if j == 4:
            #     print("ANKLE ANGLE {}".format(joint_trajectories[j][t]))

            # print("j: {}".format(j))
            action[j] = joint_trajectories[j][t]
        # rospy.logdebug("Selected Acton: {}".format(action))

        # Perform action
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward


if __name__ == '__main__':
    main()
