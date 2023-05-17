"""Q learning Agent.

Chooses the best scoring value with no thought about the future.
"""
import numpy as np
from random import randint
from agents import BaseAgent


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
        self.dirty_tiles = None
        self.episode = 0

    def process_reward(self, observation: np.ndarray, info: None | dict, reward: float, old_state: tuple, new_state: tuple, action: int):
        tile_state = []
        for dirty_tile in self.dirty_tiles:
            if observation.T[dirty_tile[0]][dirty_tile[1]] == 3:
                tile_state += [1]
            else:
                tile_state += [0]

        new_state = [new_state[0], new_state[1]] + tile_state
        new_state = tuple(new_state)

        if sum(info["dirt_cleaned"]) == 1:
            index = self.dirty_tiles.index(
                (info["agent_pos"][self.agent_number][1], info["agent_pos"][self.agent_number][0]))
            tile_state[index] = 1
        old_state = [old_state[0], old_state[1]] + tile_state
        old_state = tuple(old_state)

        if info['agent_charging'][self.agent_number] == True:
            # print("DECAY EPSILON")

            self.eps = max(0, self.eps - self.ed)
            # print("New value epsilon: ", self.eps)

        self.updateQ(old_state, new_state, action, reward)

    def updateQ(self, old_state, new_state, action, reward):
        # Bellman equation
        self.q_table[old_state][action] = (1-self.lr)*self.q_table[old_state][action] + self.lr * (
            reward + self.dr*np.max(self.q_table[new_state]))

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        # print(info)
        if self.dirty_tiles is None:
            # print(observation.T)
            self.dirty_tiles = list(zip(np.where(observation.T == 3)[
                0], np.where(observation.T == 3)[1]))
            num_of_dirty_tiles = len(self.dirty_tiles)
            # print(num_of_dirty_tiles)
            shape = [np.shape(observation)[0], np.shape(observation)[
                1]] + [2 for i in range(num_of_dirty_tiles)] + [4]
            self.q_table = np.zeros(shape)

        x, y = info["agent_pos"][self.agent_number]

        eps = np.random.uniform(0, 1)
        if eps > self.eps:

            tile_state = []
            for dirty_tile in self.dirty_tiles:
                if observation.T[dirty_tile[0]][dirty_tile[1]] == 3:
                    tile_state += [1]
                else:
                    tile_state += [0]

            new_state = [x, y] + tile_state
            new_state = tuple(new_state)

            return np.argmax(self.q_table[new_state])
        else:
            return randint(0, 3)
