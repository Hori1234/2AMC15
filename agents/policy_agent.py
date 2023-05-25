# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:50:00 2023

@author: 20183067
"""
"""Agent Base.

We define the base class for all agents in this file.
"""
from abc import ABC, abstractmethod

import numpy as np

class Policy_iteration(ABC):
    def __init__(self, agent_number, gamma):
        """Base agent. All other agents should build on this class.
        As a reminder, you are free to add more methods/functions to this class
        if your agent requires it.
        """
        self.agent_number = agent_number
        self.obs = {}

        self.states = []
        self.actions = [0,1,2,3]
        self.policy = {}
        self.prev_move = 4
        self.prev_s = 0
        self.V = 0
        self.dirty_tiles = []
        self.n_dirts = 0
        self.gamma = gamma
        self.opt = {}
        self.explore = True
        self.charging = 0
        self.next_states_dict = {}

    def process_reward(self, observation: np.ndarray, reward: float, info):

        """Any code that processes a reward given the observation is here.
        Args:
            observation: The observation which is returned by the environment.
            reward: The float value which is returned by the environment as a
                reward.
        """
        s = self.prev_s
        a = self.prev_move
        self.rewards[s, a] = reward
        self.policy = self.Policy_iteration()
        if (reward == 10) & (self.explore):
            self.explore = False
            self.charging = info['agent_pos'][0]
        return False
    def next_state(self, s:tuple,a:int):
        x = s[0]
        y = s[1]
        next_xy = {1: (x, y - 1), 0: (x, y + 1), 2: (x - 1, y), 3: (x + 1, y), 4: (x, y)}
        new_s = next_xy[a]
        if (0 < new_s[0] < len(self.obs)) & (0 < new_s[1] < len(self.obs[0])):
            return new_s
        else:
            return s
    def Policy_iteration(self):
        """"Possible actions:
            - 0: Move down
            - 1: Move up
            - 2: Move left
            - 3: Move right
            - 4: Stand still
        """
        """"Keep track of board positions and what is present at which tile
        0: empty tile
        1: unknown
        2: Wall
        3: Dirty tile
        4: Charging-station --> robot should end there
        """

        it = 0
        # Initialize actions randomly

        while True:
            old_policy = self.policy.copy()
            self.V = self.policy_evaluation()

            self.policy = self.policy_improvement()
            if all(old_policy[s] == self.policy[s] for s in self.states) or it>10: #change to theta-variable
                break
            it += 1
        return self.policy

    def policy_evaluation(self, theta = 0.5):
        #self.V = {s: 0 for s in self.states}
        max_it = 0
        delta = 0
        while True:
            oldV = self.V.copy()
            for s in self.states:
                a = self.policy[s]
                if self.next_states_dict[s, a] in self.dirty_tiles:
                    addition = -1
                elif (self.next_states_dict[s, a] == self.charging) & (len(self.dirty_tiles) != self.n_dirts):
                    addition = -5
                elif (self.next_states_dict[s, a] == self.charging) & (len(self.dirty_tiles) == self.n_dirts):
                    addition = 10
                else:
                    addition = self.rewards[s, a]
                self.V[s] = addition + self.gamma * oldV[self.next_states_dict[s, a]]

                delta = max(delta, np.abs(self.V[s] - oldV[s]))

            if delta < theta or max_it > 10:
                break
            max_it += 1
        return self.V
    def next_state(self, s:tuple,a:int):
        x = s[0]
        y = s[1]
        next_xy = {1: (x, y - 1), 0: (x, y + 1), 2: (x - 1, y), 3: (x + 1, y), 4: (x, y)}
        new_s = next_xy[a]
        if (0 < new_s[0] < len(self.obs)-1) & (0 < new_s[1] < len(self.obs[0])-1):
            return new_s
        else:
            return s

    def policy_improvement(self):
        for s in self.states:
            Q = {}
            for a in self.actions:
                if self.next_states_dict[s, a] in self.dirty_tiles:
                    addition = -1
                elif (self.next_states_dict[s, a] == self.charging) & (len(self.dirty_tiles) != self.n_dirts):
                    addition = -5
                elif (self.next_states_dict[s, a] == self.charging) & (len(self.dirty_tiles) == self.n_dirts):
                    addition = 10
                else:
                    addition = self.rewards[s, a]
                s_next = self.next_states_dict[s, a]
                Q[a] = addition + self.gamma*self.V[s_next]

            # max_value = max(Q.values())
            # max_keys = [k for k, v in Q.items() if v == max_value]
            # policy[s] = np.random.choice(max_keys)

            self.policy[s] = max(Q, key=Q.get)
        return self.policy

    # @abstractmethod
    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        """Any code that does the action should be included here.

        Args:
            observation: The observation which is returned by the environment.
            info: Any additional information your agent needs can be passed
                in here as well as a dictionary.
        """
        if not self.states:
            self.states = [(x, y) for x in range(1,len(observation)-1) for y in range(1,len(observation[0])-1) ]
            self.policy = {s: np.random.choice(self.actions) for s in self.states}
            self.obs = observation
            self.rewards = {((x, y), a): 0 for x in range(1,len(observation)-1) for y in range(1,len(observation[0])-1) for a in self.actions}
            self.next_states_dict = {(s, a): self.next_state(s, a) for s in self.states for a in self.actions}
            self.V = {s: 0 if (0 < s[0] < len(self.obs)) & (0 < s[1] < len(self.obs[0])) else -5 for s in self.states}
        if info['dirt_cleaned'][0] > 0:
            self.dirty_tiles.append(info['agent_pos'][0])
            if self.explore:
                self.n_dirts += 1
        self.policy = self.Policy_iteration()
        policy_to_choose = self.policy
        move = policy_to_choose[info['agent_pos'][0]]
        self.prev_move = move
        self.prev_s = info['agent_pos'][0]


        return move

