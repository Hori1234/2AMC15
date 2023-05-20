"""Q learning Agent.

Chooses the best scoring value with no thought about the future.
"""
import numpy as np
from random import randint
from agents import BaseAgent
from copy import copy


class QAgent(BaseAgent):
    def __init__(self, agent_number, learning_rate=0.2, discount_rate=0.99, epsilon_decay=0.01, epsilon_step_increment=0.0001):
        """Chooses an action randomly unless there is something neighboring.

        Args:
            agent_number: The index of the agent in the environment.
        """
        super().__init__(agent_number)
        self.q_table = None
        self.lr = learning_rate
        self.dr = discount_rate
        self.ed = epsilon_decay
        self.eps = 1
        self.dirty_tiles = []
        self.episode = 0
        self.initialized = False
        self.tile_state = []

    def process_reward(self, observation: np.ndarray, info: None | dict, reward: float, old_state: tuple, new_state: tuple, action: int):
        x, y = info["agent_pos"][self.agent_number]
        old_h, old_w = np.shape(self.q_table)[0], np.shape(self.q_table)[1]
        if x+1 > np.shape(self.q_table)[0]:
            temp_q_table = self.q_table
            shape = [np.shape(self.q_table)[0]+1, np.shape(self.q_table)
                     [1]] + [2 for i in range(len(self.tile_state))] + [4]
            self.q_table = np.zeros(shape)
            self.q_table[:old_h, :old_w] = temp_q_table

        elif y+1 > np.shape(self.q_table)[1]:
            temp_q_table = self.q_table
            shape = [np.shape(self.q_table)[0], np.shape(self.q_table)[
                1]+1] + [2 for i in range(len(self.tile_state))] + [4]
            self.q_table = np.zeros(shape)
            self.q_table[:old_h, :old_w] = temp_q_table

        old_tile_state = copy(self.tile_state)

        if sum(info["dirt_cleaned"]) == 1:
            # If dirty tile not yet recorded, expand the statespace and remember the dirty tile.
            if (x, y) not in self.dirty_tiles:
                old_tile_state = self.updateShapeQ(x, y, old_tile_state)

        new_state = [new_state[0], new_state[1]] + self.tile_state
        new_state = tuple(new_state)

        old_state = [old_state[0], old_state[1]] + old_tile_state
        old_state = tuple(old_state)

        # If the agent is charging, the episode has ended and update the epsilon value for the subsequent episode.
        if info['agent_charging'][self.agent_number] == True:
            self.eps = max(0, self.eps - self.ed)
            self.tile_state = [1 for i in range(len(self.tile_state))]

        # Update the Q table
        self.updateQ(old_state, new_state, action, reward)

    def updateShapeQ(self, x, y, old_tile_state):
        self.dirty_tiles += [(x, y)]
        self.tile_state += [0]
        old_tile_state += [1]
        temp_q_table = self.q_table
        shape = [np.shape(self.q_table)[0], np.shape(self.q_table)[
            1]] + [2 for i in range(len(self.tile_state))] + [4]
        self.q_table = np.zeros(shape)
        idx = [slice(None)]*self.q_table.ndim
        axis = -2
        idx[axis] = 1
        self.q_table[tuple(idx)] = temp_q_table
        return old_tile_state

    def updateQ(self, old_state, new_state, action, reward):
        # Bellman equation
        self.q_table[old_state][action] = (1-self.lr)*self.q_table[old_state][action] + self.lr * (
            reward + self.dr*np.max(self.q_table[new_state]))

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        x, y = info["agent_pos"][self.agent_number]

        if self.dirty_tiles:
            if (x, y) in self.dirty_tiles:
                index = self.dirty_tiles.index((x, y))
                self.tile_state[index] = 0

        if not self.initialized:
            self.q_table = np.zeros((x+1, y+1, 4))
            self.initialized = True

        eps = np.random.uniform(0, 1)

        if eps > self.eps:
            new_state = [x, y] + self.tile_state
            new_state = tuple(new_state)
            return np.argmax(self.q_table[new_state])
        else:
            return randint(0, 3)
