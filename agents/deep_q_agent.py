import random
from collections import namedtuple, deque
from random import randint
from copy import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from agents import BaseAgent

# if GPU is to be used
device = torch.device("cuda" if torch.backends.cuda.is_built() and torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayMemory(object):
    """
    Experience replay memory buffer for storing and sampling transitions.
    """

    def __init__(self, capacity):
        """
        Initialize the ReplayMemory object with a specified capacity.

        Args:
            capacity: The maximum capacity of the replay memory.
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """
        Save a transition to the memory.

        Args:
            *args: Variable length argument list representing a transition.
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from the memory.

        Args:
            batch_size: The size of the batch to sample.

        Returns:
            A list of randomly sampled transitions.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Return the current size of the memory.

        Returns:
            The size of the memory.
        """
        return len(self.memory)


class DQN(nn.Module):
    """
    Deep Q-Network (DQN) model.
    """

    def __init__(self, n_observations, n_actions):
        """
        Initialize the DQN model with the specified number of observations and actions.

        Args:
            n_observations: The number of observations.
            n_actions: The number of actions.
        """
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Perform forward pass through the network.

        Args:
            x: Input tensor.

        Returns:
            Output tensor representing Q-values for each action.
        """
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        return self.layer3(x)

class DeepQAgent(BaseAgent):
    """
    Deep Q-Learning agent for choosing actions based on learned Q-values.
    """
    def __init__(self, agent_number, battery_size, learning_rate=0.00002, gamma=0.8, epsilon_decay=0.001, memory_size=100000, batch_size=128, tau=0.5, epsilon_stop=0.4):
        """
        Initialize the Deep Q-Learning agent.

        Args:
            agent_number: The index of the agent in the environment.
            battery_size: The maximum battery capacity of the agent.
            learning_rate: The learning rate for the optimizer (default: 0.00002).
            gamma: The discount factor for future rewards (default: 0.8).
            epsilon_decay: The rate of decay for epsilon exploration (default: 0.001).
            memory_size: The maximum size of the replay memory (default: 100000).
            batch_size: The batch size for optimization (default: 128).
            tau: The update rate for the target network (default: 0.5).
            epsilon_stop: The minimum value of epsilon (default: 0.4).
        """
        super().__init__(agent_number)
        self.lr = learning_rate
        self.gamma = gamma
        self.ed = epsilon_decay
        self.eps = 1
        self.eps_stop = epsilon_stop
        self.tau = tau
        self.w = None
        self.h = None
        self.initialized = False
        self.dirty_tiles = []
        self.tile_state = []
        self.first_run = True
        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.battery_size = battery_size
        self.loss = 0
        self.converged = False

    def initialize_network(self, n_of_states, n_actions):
        """
        Initializes the policy network, target network, and optimizer.

        Args:
            n_of_states (int): Number of input states for the neural networks.
            n_actions (int): Number of possible actions for the agent.

        Returns:
            None
        """
        # Create the policy network and the target network.
        self.policy_net = DQN(n_of_states, n_actions).to(device)
        self.target_net = DQN(n_of_states, n_actions).to(device)
        # Copy the weights of the policy network to the target network.
        self.target_net.load_state_dict(self.policy_net.state_dict())
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
        action: int,
        old_battery_state: int,
    ):
        """
        Processes the reward, updates the agent's memory, and optimizes the model.

        Args:
            observation (np.ndarray): The current observation/state.
            info (None | dict): Additional information about the environment.
            reward (float): The reward received for the previous action.
            old_state (tuple): The previous state of the agent.
            action (int): The action taken in the previous state.
            old_battery_state (int): The previous battery state of the agent.

        Returns:
            bool: True if the agent should stop learning (converged), False otherwise.
        """
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
            clear_state[x, y] = 1
            if info["agent_charging"][self.agent_number] == False and old_battery_state == 1:
                new_battery_state = [0]
            else:
                new_battery_state = [info['battery_left'][self.agent_number]/self.battery_size]
            new_state = list(clear_state.flatten()) + self.tile_state + new_battery_state

            # Create the state vector of the previous state.
            clear_state = np.zeros((self.h, self.w), dtype=np.uint8)
            clear_state[old_state[0], old_state[1]] = 1
            old_battery_state = [old_battery_state/self.battery_size]
            old_state = list(clear_state.flatten()) + old_tile_state + old_battery_state

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

        if info["agent_charging"][self.agent_number] == True and (3 not in observation):
            # If the agent is charging and all dirty tiles have been cleaned,
            # the episode has ended and update the epsilon value for the subsequent episode.
            self.eps = max(0, self.eps - self.ed)
            # Also, all the dirty tiles are reset so the tile state should only contain 1's.
            self.tile_state = [1 for i in range(len(self.tile_state))]
            if self.first_run:
                # If this was the first run, determine the size of the input layer of the neural network.
                # The state space is increased by 1 for the battery state.
                state_space = self.w * self.h + len(self.tile_state) + 1
                # Initialize the neural network.
                self.initialize_network(state_space, 4)
                # The first run is over so set the first run boolean to False.
                self.first_run = False

        # Return true if the agent should stop learning (converged). When epsilon equals zero the agent is terminated.
        if self.eps < self.eps_stop:
            return True
        else:
            return False

    def updateTileShape(self, x, y, old_tile_state):
        """
        Updates the tile state and adds the newly found dirty tile to the list of dirty tiles.

        Args:
            x (int): The x coordinate of the agent.
            y (int): The y coordinate of the agent.
            old_tile_state (list): A binary list of which dirty tiles have and have not been cleaned yet.

        Returns:
            list: The updated binary list of the previous tile state.
        """
        # Add the new found dirty tile to the list.
        self.dirty_tiles += [(x, y)]
        # Update the tile state by adding a 0. The 0 is added because the current tile was dirty but not anymore as the robot is here currently.
        self.tile_state += [0]
        # Update the previous tile state by adding a 1 (just before this state, this particular tile was dirty because the agent cleaned it this turn).
        old_tile_state += [1]
        return old_tile_state

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        """
        Selects an action for the agent to take based on the current observation and information.

        Args:
            observation (np.ndarray): The current observation of the environment.
            info (None or dict): Additional information about the environment.

        Returns:
            int: The selected action for the agent.

        Notes:
            - The action values correspond to the following directions: 0 = up, 1 = right, 2 = down, 3 = left.
            - The method uses an epsilon-greedy approach to explore and exploit the agent's knowledge.

        """
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
            battery_state = [info['battery_left'][self.agent_number]/self.battery_size]
            state = list(state.flatten()) + self.tile_state + battery_state
            with torch.no_grad():
                # Return the action belonging to the highest value in the output of the neural network.
                return self.policy_net(torch.tensor(state).float().to(device)).max(0)[1]
        else:
            return randint(0, 3)

    def optimize_model(self):
        """
        Performs a single optimization step for the policy network based on the stored experiences.

        Notes:
            - This method implements the Deep Double Q-Network (DDQN) algorithm.
            - It uses a mini-batch of transitions sampled from the replay memory to update the policy network.
            - The target network is also updated to track the changes in the policy network gradually.

        """
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
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states
            ).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.loss = loss.item()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Update the target net to reflect the changes in the policy net (but slowly using tau).
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def update_agent(self,dirty_tales,tile_state, terminated):
        ##Update dirty_tales and tile_state
        self.dirty_tiles = copy(dirty_tales)
        self.tile_state = copy(tile_state)
        if terminated:
            self.eps = max(0, self.eps - self.ed)
            # Also, all the dirty tiles are reset so the tile state should only contain 1's.
            self.tile_state = [1 for i in range(len(self.tile_state))]
            if self.first_run:
                # If this was the first run, determine the size of the input layer of the neural network.
                # The state space is increased by 1 for the battery state.
                state_space = self.w * self.h + len(self.tile_state) + 1
                # state_space = 2+len(self.tile_state)
                # print(f'State space: {state_space}')
                # Initialize the neural network.
                self.initialize_network(state_space, 4)
                # The first run is over so set the first run boolean to False.
                self.first_run = False
        else:
            self.tile_state = copy(tile_state)
            if self.first_run:
                self.dirty_tiles = copy(dirty_tales)
