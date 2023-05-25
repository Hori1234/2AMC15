"""Agent Base.

We define the base class for all agents in this file.
"""
from abc import ABC, abstractmethod

import numpy as np


class Policy_iteration(ABC):
    def __init__(self, agent_number, gamma):
        """
        Base agent. All other agents should build on this class.

        Args:
            agent_number (int): The identifier for the agent.

        Attributes:
            agent_number (int): The identifier for the agent.
            obs (dict): Dictionary to store observations.
            states (list): List to store the states.
            actions (list): List of available actions.
            policy (dict): Dictionary to store the policy.
            prev_move (int): Previous move made by the agent.
            prev_s (int): Previous state.
            V (int): Value function.
            dirty_tiles (list): List of dirty tiles.
            n_dirts (int): Number of dirty tiles.
            gamma (float): Discount factor for future rewards.
            opt (dict): Dictionary for optimization.
            train (bool): Flag indicating if the agent is in training mode.
            charging (int): Counter for charging state.
        """
        self.agent_number = agent_number
        self.obs = {}

        self.states = []
        self.actions = [0, 1, 2, 3]
        self.policy = {}
        self.prev_move = 4
        self.prev_s = 0
        self.V = 0
        self.dirty_tiles = []
        self.n_dirts = 0
        self.gamma = gamma
        self.opt = {}
        self.train = True
        self.charging = 0

    def process_reward(self, observation: np.ndarray, reward: float, info):
        """
        Processes a reward given an observation in the environment.

        Args:
            observation (np.ndarray): The observation returned by the environment.
            reward (float): The reward returned by the environment.
            info: Additional information provided by the environment.

        Returns:
            float: The processed reward.

        Notes:
            - This function updates the rewards table based on the previous state
              and action.
            - It performs policy iteration to update the agent's policy.
            - If the reward is 10 and the agent is in training mode, it sets the
              `train` flag to False and records the agent's charging position.
        """
        s = self.prev_s
        a = self.prev_move
        self.rewards[s, a] = reward
        self.policy = self.Policy_iteration()
        if reward == 10 and self.train:
            self.train = False
            self.charging = info["agent_pos"][0]
        return False

    def Policy_iteration(self):
        """
        Performs the policy iteration algorithm to determine the optimal policy.

        Returns:
            dict: The optimal policy.

        Notes:
            - The function keeps track of board positions and what is present at
              each tile using the following codes:
              * 0: empty tile
              * 1: unknown
              * 2: Wall
              * 3: Dirty tile
              * 4: Charging-station (robot should end there)
            - The policy iteration algorithm consists of two main steps: policy
              evaluation and policy improvement.
            - The algorithm iteratively updates the policy until convergence or
              a maximum number of iterations is reached.
            - The algorithm breaks if the new policy is the same as the previous
              policy or the maximum number of iterations is exceeded.
        """

        it = 0

        while True:
            old_policy = self.policy.copy()
            self.V = self.policy_evaluation()

            self.policy = self.policy_improvement()
            if (
                all(old_policy[s] == self.policy[s] for s in self.states) or it > 5
            ):  # change to theta-variable
                break
            it += 1
        return self.policy

    def policy_evaluation(self, theta=0.5):
        """
        Performs policy evaluation to estimate the value function for the current policy.

        Args:
            theta (float): The threshold value for convergence (default: 0.5).

        Returns:
            np.ndarray: The updated value function.

        Notes:
            - This function iteratively updates the value function until convergence
              or a maximum number of iterations is reached.
            - The value function is updated for each state based on the current policy.
            - The updates take into account the rewards, transition probabilities, and
              the discount factor.
            - The convergence is determined by the maximum change in the value function
              (delta) being below the given threshold (theta) or reaching the maximum
              number of iterations allowed.
        """

        max_it = 0
        delta = 0
        while True:
            oldV = self.V.copy()
            for s in self.states:
                a = self.policy[s]
                if self.next_state(s, a) in self.dirty_tiles:
                    addition = -1
                elif (self.next_state(s, a) == self.charging) & (
                    len(self.dirty_tiles) != self.n_dirts
                ):
                    addition = -5
                elif (self.next_state(s, a) == self.charging) & (
                    len(self.dirty_tiles) == self.n_dirts
                ):
                    addition = 10
                else:
                    addition = self.rewards[s, a]
                if addition == -5:
                    self.V[s] = addition
                else:
                    self.V[s] = addition + self.gamma * oldV[self.next_state(s, a)]
                self.V[s] = addition + self.gamma * oldV[self.next_state(s, a)]

                delta = max(delta, abs(self.V[s] - oldV[s]))
            if delta < theta or max_it > 5:
                break
            max_it += 1
        return self.V

    def next_state(self, s: tuple, a: int):
        """
        Computes the next state given the current state and action.

        Args:
            s (tuple): The current state as a tuple (x, y).
            a (int): The action to be taken.

        Returns:
            tuple: The next state as a tuple (x, y).

        Notes:
            - The function determines the next state based on the current state (s)
              and the action (a) to be taken.
            - The new state (new_s) is computed based on the current coordinates (x, y)
              and the action's corresponding direction.
            - If the new state falls within the boundaries of the observation space,
              it is returned as the next state; otherwise, the current state is returned.
        """
        x = s[0]
        y = s[1]
        next_xy = {
            1: (x, y - 1),
            0: (x, y + 1),
            2: (x - 1, y),
            3: (x + 1, y),
            4: (x, y),
        }
        new_s = next_xy[a]
        if (0 < new_s[0] < len(self.obs) - 1) & (0 < new_s[1] < len(self.obs[0]) - 1):
            return new_s
        else:
            return s

    def policy_improvement(self):
        """
        Performs policy improvement to update the agent's policy.

        Returns:
            dict: The updated policy.

        Notes:
            - This function iterates over all states and actions to compute the
              Q-values for each action.
            - The Q-values are determined based on the rewards, transition probabilities,
              and the discount factor.
            - The function updates the policy by selecting the action that maximizes
              the Q-value for each state.
            - The updated policy is returned.
        """
        for s in self.states:
            Q = {}
            for a in self.actions:
                if self.next_state(s, a) in self.dirty_tiles:
                    addition = -1
                elif (self.next_state(s, a) == self.charging) & (
                    len(self.dirty_tiles) != self.n_dirts
                ):
                    addition = -5
                elif (self.next_state(s, a) == self.charging) & (
                    len(self.dirty_tiles) == self.n_dirts
                ):
                    addition = 10
                else:
                    addition = self.rewards[s, a]
                if addition == -5:
                    Q[a] = addition
                else:
                    s_next = self.next_state(s, a)
                    Q[a] = addition + self.gamma * self.V[s_next]

            self.policy[s] = max(Q, key=Q.get)
        return self.policy

    # @abstractmethod
    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        """
        Performs the action based on the current observation and agent's policy.

        Args:
            observation (np.ndarray): The observation returned by the environment.
            info (None or dict): Additional information needed by the agent, if available.

        Returns:
            int: The selected action to be taken.

        Notes:
            - This function includes the code responsible for taking the action.
            - If the agent's states list is empty, it initializes various attributes
              such as states, policy, rewards, next_states_dict, and V based on the
              given observation.
            - If any dirt is cleaned, it updates the list of dirty_tiles and the count
              of cleaned dirts (n_dirts) if exploration is enabled.
            - The agent's policy is then updated using the policy iteration algorithm.
            - The action to be taken is determined based on the agent's policy and the
              current agent position (agent_pos).
            - The selected action is returned.
        """
        if not self.states:
            self.states = [
                (x, y)
                for x in range(1, len(observation) - 1)
                for y in range(1, len(observation[0]) - 1)
            ]
            self.policy = {s: np.random.choice(self.actions) for s in self.states}
            self.obs = observation
            self.rewards = {
                ((x, y), a): 0
                for x in range(1, len(observation) - 1)
                for y in range(1, len(observation[0]) - 1)
                for a in self.actions
            }
            self.V = {
                s: 0
                if (0 < s[0] < len(self.obs)) & (0 < s[1] < len(self.obs[0]))
                else -5
                for s in self.states
            }
        if info["dirt_cleaned"][0] > 0:
            self.dirty_tiles.append(info["agent_pos"][0])
            if self.train:
                self.n_dirts += 1
        self.policy = self.Policy_iteration()
        policy_to_choose = self.policy
        move = policy_to_choose[info["agent_pos"][0]]
        self.prev_move = move
        self.prev_s = info["agent_pos"][0]

        return move
