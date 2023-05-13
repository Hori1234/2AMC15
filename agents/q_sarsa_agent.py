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
        self.epsilon_decay = 0.001
        self.max_epsilon = 1
        self.min_epsilon = 0.01
        self.gamma = gamma
        self.reward = 0
        self.previous_state = [None] * 3
        # Initialize observation
        self.observation = obs
        # Initialize actions
        self.actions = {"UP": 0, "DOWN": 1, "RIGHT": 2, "LEFT": 3, "STUCK": 4}
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

    # generate_q_table_rewards
    def generate_q_table_rewards(self, observation: np.ndarray):
        """
        Generates a Q-table with all zeros.
        And with rewards for obstacles, dirt, and home

        Args:
            shape: The shape of the Q-table.
        """
        # initialize q_table with all zeros
        q_table = np.zeros(observation.shape).astype(float)
        # set q_table values for obstacles, dirt, and home
        for i in range(observation.shape[0]):
            for j in range(observation.shape[1]):
                # set q_table values for obstacles, dirt, and home
                if observation[i][j] == 2 or observation[i][j] == 1:
                    q_table[i][j] = 1
                elif observation[i][j] == 3:
                    q_table[i][j] = 1
                elif observation[i][j] == 4:
                    q_table[i][j] = 1
                elif observation[i][j] == 0:
                    q_table[i][j] = 5

        return q_table

    # heuristic_distance_to_dirt
    def heuristic_distance_to_dirt(grid: Grid, info: None | dict) -> float:
        """Count the number of tiles cleand by the agent.

        Args:
            grid: The grid object.
            info: The info dictionary returned by the environment.

        Returns:
            The number of tiles cleaned by the agent.
        """
        obs = grid.cells

        def calculate_distance(current_state, dirt_pos):
            x1, y1 = current_state
            x2, y2 = dirt_pos
            distance = abs(x1 - x2) + abs(y1 - y2)
            return distance

        # Calculate the distance to the nearest dirt tile, avoiding obstacles
        current_state = info["agent_pos"][0]
        empty_positions = [
            (x, y) for x, y in zip(np.where(obs == 0)[0], np.where(obs == 0)[1])
        ]
        dirt_positions = [
            (x, y) for x, y in zip(np.where(obs == 3)[0], np.where(obs == 3)[1])
        ]
        obstacle_positions = [
            (x, y) for x, y in zip(np.where(obs == 2)[0], np.where(obs == 2)[1])
        ]

        if len(dirt_positions) == 0:
            charging_station_positions = (
                np.where(obs == 4)[0][0],
                np.where(obs == 4)[1][0],
            )

            # No dirt tiles remaining, return a high reward
            dirt_positions.append(charging_station_positions)

        min_distance = float("inf")
        for dirt_pos in dirt_positions:
            if dirt_pos in obstacle_positions:
                continue
            distance = calculate_distance(current_state, dirt_pos)
            min_distance = min(min_distance, distance)

        # Adjust the distance by the number of obstacles along the path
        path_obstacles = set(obstacle_positions) - set(dirt_positions)
        num_obstacles = len(path_obstacles)
        adjusted_distance = min_distance + num_obstacles

        return 1.0 / adjusted_distance if adjusted_distance != float("inf") else 0.0

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
        self.update_DQ(x, y, action, reward, next_x, next_y)
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
        self, x: int, y: int, action_idx: int, reward: int, next_x: int, next_y: int
    ) -> None:
        if self.use_double_q == True:
            if np.random.rand() < 0.5:
                q_table = self.q_tableA
            else:
                q_table = self.q_tableB
            next_max = np.max(q_table[:, next_x, next_y])
        else:
            next_max = np.max(self.q_tableA[:, next_x, next_y])
        # Get the maximum Q-value for the next state
        if self.use_sarsa:
            # Get the next action
            next_action = self.get_next_best_move(next_x, next_y)
            # Use the next action to calculate the next Q-value using SARSA
            current_value = self.q_tableA[next_action, next_x, next_y]
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
        self.q_table[action_idx][x][y] = new_value

    def get_next_move(self, x: int, y: int) -> tuple[int, int, int]:
        next_x = x
        next_y = y
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
        action_idx = np.argmax(self.q_table[:, x, y])
        # print(action_idx)
        # for k,v in self.actions.items():
        #     if v == action_idx:
        #         f = open("./results/path.txt", "a")
        #         #print("Action: ", k, "Action Index: ", action_idx, "Q-Table: ", self.q_table[:,x,y], "X: ", x, "Y: ", y)
        #         f.write("Action: " + str(k) + " Action Index: " + str(action_idx) + " Q-Table: " + str(self.q_table[:,x,y]) + " X: " + str(x) + " Y: " + str(y))
        #         f.write("\n")
        #         f.close()
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
