import copy
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import gym
import sys

# Twin Delayed Deterministic Policy Gradient

# Algorithm Steps

# 1. Initiailize Networks


class Actor(nn.Module):
    """Initialize parameters and build model.
       An nn.Module contains layers, and a method
       forward(input)that returns the output.
       Weights (learnable params) are inherently defined here.

        Args:
            state_dim (int): Dimension of each state
            action_dim (int): Dimension of each action
            max_action (float): highest action to take

        Return:
            action output of network with tanh activation
    """
    def __init__(self, state_dim, action_dim, max_action):
        # Super calls the nn.Module Constructor
        super(Actor, self).__init__()
        # input layer
        self.fc1 = nn.Linear(state_dim, 400)
        # hidden layer
        self.fc2 = nn.Linear(400, 300)
        # output layer
        self.fc3 = nn.Linear(300, action_dim)
        # wrap from -max to +max
        self.max_action = max_action

    def forward(self, state):
        # You just have to define the forward function,
        # and the backward function (where gradients are computed)
        # is automatically defined for you using autograd.
        # Learnable params can be accessed using Actor.parameters

        # Here, we create the tensor architecture
        # state into layer 1
        a = F.relu(self.fc1(state))
        # layer 1 output into layer 2
        a = F.relu(self.fc2(a))
        # layer 2 output into layer 3 into tanh activation
        return self.max_action * torch.tanh(self.fc3(a))


class Critic(nn.Module):
    """Initialize parameters and build model.
        Args:
            state_dim (int): Dimension of each state
            action_dim (int): Dimension of each action

        Return:
            value output of network
    """
    def __init__(self, state_dim, action_dim):
        # Super calls the nn.Module Constructor
        super(Critic, self).__init__()

        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

        # Q2 architecture
        self.fc4 = nn.Linear(state_dim + action_dim, 400)
        self.fc5 = nn.Linear(400, 300)
        self.fc6 = nn.Linear(300, 1)

    def forward(self, state, action):
        # concatenate state and actions by adding rows
        # to form 1D input layer
        sa = torch.cat([state, action], 1)

        # s,a into input layer into relu activation
        q1 = F.relu(self.fc1(sa))
        # l1 output into l2 into relu activation
        q1 = F.relu(self.fc2(q1))
        # l2 output into l3
        q1 = self.fc3(q1)

        # s,a into input layer into relu activation
        q2 = F.relu(self.fc4(sa))
        # l4 output into l5 into relu activation
        q2 = F.relu(self.fc5(q2))
        # l5 output into l6
        q2 = self.fc6(q2)
        return q1, q2

    def Q1(self, state, action):
        # Return Q1 for gradient Ascent on Actor
        # Note that only Q1 is used for Actor Update

        # concatenate state and actions by adding rows
        # to form 1D input layer
        sa = torch.cat([state, action], 1)

        # s,a into input layer into relu activation
        q1 = F.relu(self.fc1(sa))
        # l1 output into l2 into relu activation
        q1 = F.relu(self.fc2(q1))
        # l2 output into l3
        q1 = self.fc3(q1)
        return q1


# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    """Buffer to store tuples of experience replay"""
    def __init__(self, max_size=1000000):
        """
        Args:
            max_size (int): total amount of tuples to store
        """

        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        """Add experience tuples to buffer

        Args:
            data (tuple): experience replay tuple
        """

        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        """Samples a random amount of experiences from buffer of batch size

        Args:
            batch_size (int): size of sample
        """

        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, next_states, rewards, dones = [], [], [], [], []

        for i in ind:
            s, a, s_, r, d = self.storage[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            next_states.append(np.array(s_, copy=False))
            rewards.append(np.array(r, copy=False))
            dones.append(np.array(d, copy=False))

        return np.array(states), np.array(actions), np.array(
            next_states), np.array(rewards).reshape(
                -1, 1), np.array(dones).reshape(-1, 1)


class TD3Agent(object):
    """Agent class that handles the training of the networks and
       provides outputs as actions

        Args:
            state_dim (int): state size
            action_dim (int): action size
            max_action (float): highest action to take
            device (device): cuda or cpu to process tensors
            env (env): gym environment to use
            batch_size(int): batch size to sample from replay buffer
            discount (float): discount factor
            tau (float): soft update for main networks to target networks

    """
    def __init__(self,
                 env,
                 state_dim,
                 action_dim,
                 max_action,
                 discount=0.99,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=2):

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0
        self.env = env

    def select_action(self, state, noise=0.1):
        """Select an appropriate action from the agent policy

            Args:
                state (array): current state of environment
                noise (float): how much noise to add to actions

            Returns:
                action (float): action clipped within action range

        """
        # Turn float value into a CUDA Float Tensor
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise != 0:
            action = (action + np.random.normal(
                0, noise, size=self.env.action_space.shape[0]))

        return action.clip(self.env.action_space.low,
                           self.env.action_space.high)

    def train(self, replay_buffer, batch_size=100):
        """Train and update actor and critic networks

            Args:
                replay_buffer (ReplayBuffer): buffer for experience replay
                batch_size(int): batch size to sample from replay buffer\

            Return:
                actor_loss (float): loss from actor network
                critic_loss (float): loss from critic network

        """
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(
            batch_size)

        with torch.no_grad():
            """
            Autograd: If you set its attribute .requires_grad as True,
            (DEFAULT)
            it starts to track all operations on it. When you finish your
            computation you can call .backward() and have all the gradients
            computed automatically. The gradient for this tensor will be
            accumulated into .grad attribute.

            To prevent tracking history (and using memory), you can wrap
            the code block in with torch.no_grad():. This can be particularly
            helpfulwhen evaluating a model because the model may have
            trainable parameters with requires_grad=True (DEFAULT),
            but for which we don’t need the gradients

            Here, we don't want to track the acyclic graph's history
            when getting our next action because we DON'T want to train
            our actor in this step. We train our actor ONLY when we perform
            the periodic policy update. Could have done .detach() at
            target_Q = reward + not_done * self.discount * target_Q
            for the same effect
            """

            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        # A loss function takes the (output, target) pair of inputs,
        # and computes a value that estimates how far away the output
        # is from the target.
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        # Optimize the critic
        # Zero the gradient buffers of all parameters
        self.critic_optimizer.zero_grad()
        # Backprops with random gradients
        # When we call loss.backward(), the whole graph is differentiated
        # w.r.t. the loss, and all Tensors in the graph that has
        # requires_grad=True (DEFAULT) will have their .grad Tensor
        # accumulated with the gradient.
        critic_loss.backward()
        # Does the update
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            # Zero the gradient buffers
            self.actor_optimizer.zero_grad()
            # Differentiate the whole graph wrt loss
            actor_loss.backward()
            # Does the update
            self.actor_optimizer.step()

            # Update target networks (Critic 1, Critic 2, Actor)
            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data +
                                        (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data +
                                        (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),
                   filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer"))


def evaluate_policy(policy, env, eval_episodes=100, render=False):
    """run several episodes using the best agent policy

        Args:
            policy (agent): agent to evaluate
            env (env): gym environment
            eval_episodes (int): how many test episodes to run
            render (bool): show training

        Returns:
            avg_reward (float): average reward over the number of evaluations

    """

    avg_reward = 0.
    for i in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            action = policy.select_action(np.array(obs), noise=0)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("\n---------------------------------------")
    print("Evaluation over {:d} episodes: {:f}".format(eval_episodes,
                                                       avg_reward))
    print("---------------------------------------")
    return avg_reward


def trainer():
    """
    """