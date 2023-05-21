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
        self.initialized = False
        self.tile_state = []

    def process_reward(self, observation: np.ndarray, info: None | dict, reward: float, old_state: tuple, new_state: tuple, action: int):
        # Get the agents position.
        x, y = info["agent_pos"][self.agent_number]

        # If a new x coordinate is discovered, add a new column to the Q table
        if x+1 > np.shape(self.q_table)[0]:
            self.q_table = self.add_zero_column(self.q_table)

        # If a new y coordinate is discovered, add a new row to the Q table
        elif y+1 > np.shape(self.q_table)[1]:
            self.q_table = self.add_zero_row(self.q_table)

        # Copy the old tile state to a new variable
        old_tile_state = copy(self.tile_state)

        # If a tile is cleaned and the dirty tile not yet recorded, expand the statespace and remember the dirty tile.
        if sum(info["dirt_cleaned"]) == 1 and (x, y) not in self.dirty_tiles:
            old_tile_state = self.updateShapeQ(x, y, old_tile_state)

        # Get the right index for the Q table for the new state.
        new_state = [new_state[0], new_state[1]] + self.tile_state
        new_state = tuple(new_state)

        # Get the right index for the Q table for the old state.
        old_state = [old_state[0], old_state[1]] + old_tile_state
        old_state = tuple(old_state)

        # If the agent is charging, the episode has ended and update the epsilon value for the subsequent episode.
        # Also, all the dirty tiles are reset so the tile state should only contain 1's.
        if info['agent_charging'][self.agent_number] == True:
            self.eps = max(0, self.eps - self.ed)
            self.tile_state = [1 for i in range(len(self.tile_state))]

        # Update the Q table
        self.updateQ(old_state, new_state, action, reward)

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        # Get the agents position.
        x, y = info["agent_pos"][self.agent_number]

        # If the list of remembered dirty tiles is not empty, check if the current tile is one of the dirty tiles.
        # If so, find the index of the tile in the list and update the tile state.
        if self.dirty_tiles:
            if (x, y) in self.dirty_tiles:
                index = self.dirty_tiles.index((x, y))
                self.tile_state[index] = 0

        # If the Q table is not yet initialized, set the Q table with witdh and height according the current position.
        if not self.initialized:
            self.q_table = np.zeros((x+1, y+1, 4))
            self.initialized = True

        # sample a random float between 0 and 1.
        eps = np.random.uniform(0, 1)

        # If the sample is bigger than epsilon, exploit the Q table, otherwise return a random move.
        if eps > self.eps:
            new_state = [x, y] + self.tile_state
            new_state = tuple(new_state)
            return np.argmax(self.q_table[new_state])
        else:
            return randint(0, 3)


    def updateShapeQ(self, x, y, old_tile_state):
        # Add the new found dirty tile to the list.
        self.dirty_tiles += [(x, y)]
        # Update the tile state by adding a 0. The 0 is added because the current tile was dirty but not anymore as the robot is here currently.
        self.tile_state += [0]
        # Update the previous tile state by adding a 1.
        old_tile_state += [1]
        # Add an axis to the Q table for the new found dirty tile.
        self.q_table = self.add_axis(self.q_table)
        return old_tile_state

    def updateQ(self, old_state, new_state, action, reward):
        # Update the Q table using the Bellman equation.
        self.q_table[old_state][action] = (1-self.lr)*self.q_table[old_state][action] + self.lr * (
            reward + self.dr*np.max(self.q_table[new_state]))

    def add_zero_column(self, array):
        """This function adds an array of zeros to the first axis.

        Args:
            array (numpy array): The array to which the zeros will be appended.

        Returns:
            numpy array: Returns the array with the appended zeros.
        """
        return np.concatenate((array, [np.zeros(np.shape(array)[1:])]), axis=0)
        

    def add_zero_row(self, array):
        """This function adds an array of zeros to the second axis.

        Args:
            array (numpy array): The array to which the zeros will be appended.

        Returns:
            numpy array: Returns the array with the appended zeros.
        """
        array = np.moveaxis(array, [0], [1])
        array = np.concatenate((array, [np.zeros(np.shape(array)[1:])]), axis=0)
        array = np.moveaxis(array, [0], [1])
        return array

    def add_axis(self, array):
        """This function adds an axis of an array as the second to last axis.

        Args:
            array (numpy array): The array to which the axis will be added.

        Returns:
            numpy array: Returns the array with the appended axis.
        """
        result = np.array([np.zeros(np.shape(array)), array])
        return np.moveaxis(result, [0], [-2])

    