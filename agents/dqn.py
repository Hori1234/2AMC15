import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from random import randint
from copy import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 80)
        # self.layer2 = nn.Linear(80, 40)
        self.layer3 = nn.Linear(80, n_actions)
        self.dropout = nn.Dropout(0.5)

    # Called with either one element to determine next action, or a batch
    # during optimization.
    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        # x = self.dropout(x)
        # x = F.leaky_relu(self.layer2(x))
        return self.layer3(x)


"""Null Agent.

An agent which does nothing.
"""
import numpy as np

from agents import BaseAgent


class ODeepQAgent(BaseAgent):
    def __init__(
        self,
        agent_number,
        learning_rate=0.01,
        gamma=0.95,
        epsilon_decay=0.01,
        memory_size=1000,
        batch_size=100,
        tau=0.05,
    ):
        """Chooses an action based on learned q learning policy.

        Args:
            agent_number: The index of the agent in the environment.
        """
        super().__init__(agent_number)
        self.lr = learning_rate
        self.gamma = gamma
        self.ed = epsilon_decay
        self.eps = 1
        self.tau = tau
        self.w = None
        self.h = None
        self.initialized = False
        self.dirty_tiles = []
        self.tile_state = []
        self.first_run = True
        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size

    def initialize_network(self, n_of_states, n_actions):
        # Create the policy network and the target network.
        self.policy_net = DQN(n_of_states, n_actions).to(device)
        # Initialize the optimizer.
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self.lr, amsgrad=True
        )

    def process_reward(
        self,
        observation: np.ndarray,
        info: None | dict,
        reward: float,
        old_state: tuple,
        new_state: tuple,
        action: int,
    ):
        # Get the agents position.
        x, y = info["agent_pos"][self.agent_number]

        # Copy the old tile state to a new variable
        old_tile_state = copy(self.tile_state)

        if sum(info["dirt_cleaned"]) == 1 and (x, y) not in self.dirty_tiles:
            # If the agent cleans a dirt tile which it has not yet saved in its memory (self.dirty_tiles), run updateTileState().
            old_tile_state = self.updateTileShape(x, y, old_tile_state)

        if not self.first_run:
            # Create the state vector of the current state.
            clear_state = np.zeros((self.h, self.w), dtype=np.uint8)
            clear_state[new_state[0], new_state[1]] = 1
            new_state = list(clear_state.flatten()) + self.tile_state

            # Create the state vector of the previous state.
            clear_state = np.zeros((self.h, self.w), dtype=np.uint8)
            clear_state[old_state[0], old_state[1]] = 1
            old_state = list(clear_state.flatten()) + old_tile_state

            # Turn the variables into tensors for the neural network.
            old_state = torch.tensor(
                old_state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            new_state = torch.tensor(
                new_state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            reward = torch.tensor([reward], device=device)
            action = torch.tensor([[action]], device=device, dtype=torch.long)

            # Push the tuple into the memory of the agent.
            self.memory.push(old_state, action, new_state, reward)

            # Optimize the model.
            self.optimize_model()

        if info["agent_charging"][self.agent_number] == True:
            # If the agent is charging, the episode has ended and update the epsilon value for the subsequent episode.
            self.eps = max(0, self.eps - self.ed)
            # Also, all the dirty tiles are reset so the tile state should only contain 1's.
            self.tile_state = [1 for i in range(len(self.tile_state))]
            if self.first_run:
                # If this was the first run, determine the size of the input layer of the neural network.
                state_space = self.w * self.h + len(self.tile_state)
                # state_space = 2+len(self.tile_state)
                # Initialize the neural network.
                self.initialize_network(state_space, 4)
                # The first run is over so set the first run boolean to False.
                self.first_run = False

        # Return true if the agent should stop learning (converged). When epsilon equals zero the agent is terminated.
        if self.eps == 0:
            return True
        else:
            return False

    def updateTileShape(self, x, y, old_tile_state):
        """A function that adds the newly found dirty tile to the self.dirty_tiles list and updates the tile state accordingly.

        Args:
            x (int): The x coordinate of the agent.
            y (int): The y coordinate of the agent.
            old_tile_state (list): A binary list of which dirty tiles have and have not been cleaned yet.

        Returns:
            list: The binary list of the previous tile state.
        """
        # Add the new found dirty tile to the list.
        self.dirty_tiles += [(x, y)]
        # Update the tile state by adding a 0. The 0 is added because the current tile was dirty but not anymore as the robot is here currently.
        self.tile_state += [0]
        # Update the previous tile state by adding a 1 (just before this state, this particular tile was dirty because the agent cleaned it this turn).
        old_tile_state += [1]
        return old_tile_state

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        # Get the agents position.
        x, y = info["agent_pos"][self.agent_number]

        # If the agent is not yet initialized, determine the height and width of the grid.
        if not self.initialized:
            self.h, self.w = np.shape(observation)[0], np.shape(observation)[1]

        # If the list of remembered dirty tiles is not empty, check if the current tile is one of the dirty tiles.
        # If so, find the index of the tile in the list and update the tile state.
        if self.dirty_tiles:
            if (x, y) in self.dirty_tiles:
                index = self.dirty_tiles.index((x, y))
                self.tile_state[index] = 0

        # sample a random float between 0 and 1.
        eps = np.random.uniform(0, 1)

        # If the sample is bigger than epsilon, exploit the neural network, otherwise return a random move.
        if eps > self.eps:
            # Create the input layer of the neural network.
            state = np.zeros((self.h, self.w), dtype=np.uint8)
            state[x][y] = 1
            state = list(state.flatten()) + self.tile_state
            # state = [x]+[y]+self.tile_state
            with torch.no_grad():
                # Return the action belonging to the highest value in the output of the neural network.
                return self.policy_net(torch.tensor(state).float()).max(0)[1]
        else:
            return randint(0, 3)

    def optimize_model(self):
        # If the memory does not yet contain enough samples for the batch to be full, return to avoid a update of the neural network.
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of transition tuples from the memory.
        transitions = self.memory.sample(self.batch_size)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.policy_net(
                non_final_next_states
            ).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def save_model(self, path):
        print("model_saved")
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.q_network.eval()
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
