"""Agent Base.

We define the base class for all agents in this file.
"""

import numpy as np
from agents import BaseAgent
from world.grid import Grid


class QSARSA_Agent(BaseAgent):
    def __init__(
        self,
        agent_number,
        obs,
        use_sarsa=False,
        use_double_q=False,
        show_info=False,
        alpha=0.1,
        epsilon=1,
        gamma=0.9,
    ):
        """Base agent. All other agents should build on this class.

        As a reminder, you are free to add more methods/functions to this class
        if your agent requires it.
        """
        super().__init__(agent_number)
        self.agent_number = agent_number
        # Initialize hyperparameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = 0.0005
        self.max_epsilon = 1
        self.min_epsilon = 0.001
        self.gamma = gamma
        self.reward = 0
        self.previous_state = [None] * 3
        # Initialize observation
        self.observation = obs
        # Initialize actions
        self.actions = {"UP": 1, "DOWN": 0, "RIGHT": 3, "LEFT": 2, "STUCK": 4}
        self.move_list = []
        # Initialize show_info and train_range
        self.show_info = show_info
        self.home = np.where(obs == 4)
        # Initialize Q-table with all zeros
        self.q_table = self.generate_q_table(obs)
        self.q_tableA = self.generate_q_table(obs)
        self.q_tableB = self.generate_q_table(obs)
        self.use_sarsa = use_sarsa
        self.use_double_q = use_double_q

    # generate_q_table
    def generate_q_table(self, observation: np.ndarray):
        """
        Generates a Q-table with all zeros.

        Args:
            shape: The shape of the Q-table.
        """
        # initialize q_table with all zeros
        return np.zeros(
            (len(self.actions), observation.shape[0], observation.shape[1])
        ).astype(float)

    def heuristic(grid: Grid, info: None | dict) -> float:
        if info["agent_moved"][0] == False:
            return float(-100)
        elif sum(info["dirt_cleaned"]) < 1:
            return float(-1)
        elif info["agent_charging"][0] == True:
            return float(20)
        else:
            return float(5)

    def process_reward(
        self,
        observation: np.ndarray,
        reward: float,
        episode: int,
        x: float,
        y: float,
        next_x: float,
        next_y: float,
        action: float,
    ) -> None:
        """Any code that processes a reward given the observation is here.

        Args:
            observation: The observation which is returned by the environment.
            reward: The float value which is returned by the environment as a
                reward.
        """
        if self.use_double_q:
            # update the Q-table with the new Q-values using double Q-learning
            self.update_DQ(x, y, action, reward, next_x, next_y)
        else:
            # update the Q-table with the new Q-values using Q-learning
            self.update_Q(x, y, action, reward, next_x, next_y)
        # update the epsilon value
        self.epsilon = self.min_epsilon + (
            self.max_epsilon - self.min_epsilon
        ) * np.exp(-self.epsilon_decay * episode)

    def take_action(
        self, observation: np.ndarray, info: None | dict
    ) -> tuple[int, int, int]:
        """Any code that does the action should be included here.

        Args:
            observation: The observation which is returned by the environment.
            info: Any additional information your agent needs can be passed
                in here as well as a dictionary.
        """
        # get x and y coordinates of agent
        p_x, p_y = info["agent_pos"][self.agent_number]
        if self.show_info:
            self.print_state_info()
        # print(p_x, p_y, observation[p_x][p_y])
        self.previous_state = self.get_next_move(p_x, p_y)
        # Q-Learning
        return self.previous_state[0], self.previous_state[1], self.previous_state[2]

    def update_Q(
        self, x: int, y: int, action_idx: int, reward: int, next_x: int, next_y: int
    ) -> None:
        # Get the maximum Q-value for the next state
        if self.use_sarsa:
            # Get the next action
            next_action = self.get_next_best_move(next_x, next_y)
            # Use the next action to calculate the next Q-value using SARSA
            next_max = self.q_table[next_action, next_x, next_y]
        else:
            # Use the best action to calculate the next Q-value usign Q-Learning
            next_max = np.max(self.q_table[:, next_x, next_y])
        # Calculate the new Q-value
        current_value = self.q_table[action_idx][x][y]
        # Update the Q-value
        new_value = (1 - self.alpha) * current_value + self.alpha * (
            reward + self.gamma * next_max
        )
        self.q_table[action_idx][x][y] = new_value

    def update_DQ(
        self, x: int, y: int, action_idx: int, reward: float, next_x: int, next_y: int
    ) -> None:
        if np.random.rand() < 0.5:
            q_table = self.q_tableA
        else:
            q_table = self.q_tableB

        next_max = np.max(q_table[:, next_x, next_y])
        # Get the maximum Q-value for the next state
        if self.use_sarsa:
            # Get the next action
            next_action = self.get_next_best_move(next_x, next_y)
            # Use the next action to calculate the next Q-value using SARSA
            current_value = q_table[next_action, next_x, next_y]
        else:
            # Use the best action to calculate the next Q-value usign Q-Learning
            current_value = (
                self.q_tableA[action_idx, next_x, next_y]
                + self.q_tableB[action_idx, next_x, next_y]
            )
        # Update the Q-value
        new_value = (1 - self.alpha) * current_value + self.alpha * (
            reward + self.gamma * next_max
        )
        q_table[action_idx][x][y] = new_value

    def get_next_move(self, x: int, y: int) -> tuple[int, int, int]:
        next_x, next_y = x, y
        # Choose a random action with probability epsilon
        if np.random.uniform(0, 1) < self.epsilon:
            # Choose a random action with probability epsilon
            # print("Random action")
            action = self.actions[
                np.random.choice(["UP", "DOWN", "RIGHT", "LEFT", "STUCK"])
            ]
        else:
            # Choose the best action with probability 1 - epsilon
            # print("Best action")
            action = self.get_next_best_move(x, y)
        # Move the agent and get the reward
        if action == self.actions["UP"]:
            next_x, next_y = x, y - 1
        elif action == self.actions["DOWN"]:
            next_x, next_y = x, y + 1
        elif action == self.actions["LEFT"]:
            next_x, next_y = x - 1, y
        elif action == self.actions["RIGHT"]:
            next_x, next_y = x + 1, y

        # If the agent wants to hits a wall, add a penalty of -1
        return next_x, next_y, action

    def get_next_best_move(self, x: int, y: int) -> int:
        # Choose a random action with probability epsilon
        if self.use_double_q == True:
            q_values = self.q_tableA[:, x, y] + self.q_tableB[:, x, y]
            action_idx = np.argmax(q_values)
        else:
            action_idx = np.argmax(self.q_table[:, x, y])
        return action_idx

    def get_current_position(self, info: None | dict) -> float:
        return info["agent_pos"][self.agent_number]

    def print_state_info(self) -> None:
        print(self.q_table)
        print(self.observation)
        print("Current reward value:", self.reward)
        print("Current alpha value:", self.alpha)
        print("Current epsilon value:", self.epsilon)
        print("Current gamma value:", self.gamma)