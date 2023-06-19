"""Random Agent.

This is an agent that takes a random action from the available action space.
"""
from random import randint
import numpy as np

from agents import BaseAgent


class RandomAgent(BaseAgent):
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
        self.gamma = gamma
        self.tile_state = []
        self.converged = False

    """Agent that performs a random action every time. """

    def process_reward(
        self,
        observation: np.ndarray,
        info: None | dict,
        reward: float,
        old_state: tuple,
        new_state: tuple,
        action: int,
        old_battery_state: int,
        terminated: bool,
    ):
        pass

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        return randint(0, 4)
