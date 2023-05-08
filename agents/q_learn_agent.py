"""Agent Base.

We define the base class for all agents in this file.
"""
from abc import ABC, abstractmethod

import numpy as np
from agents import BaseAgent
from random import randint

class QLearn_Agent(BaseAgent):
    def __init__(self, agent_number, obs, show_info=False,train_range=10,alpha=0.2, epsilon=0.2, gamma=0.9):
        """Base agent. All other agents should build on this class.

        As a reminder, you are free to add more methods/functions to this class
        if your agent requires it.
        """
        super().__init__(agent_number)
        self.agent_number = agent_number
        # Initialize hyperparameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.reward = 0
        # Initialize observation
        self.observation = obs
        # Initialize actions
        self.actions = {"UP": 0, "DOWN": 1, "RIGHT": 2, "LEFT": 3}
        # Initialize show_info and train_range
        self.show_info = show_info
        self.train_range = train_range
        self.home = np.where(obs == 4)
        self.dirts = [(x, y) for x, y in zip(np.where(obs == 3)[0],np.where(obs == 3)[1])]
        self.obstacles = [(x, y) for x, y in zip(np.where(obs == 2)[0],np.where(obs == 2)[1])]
        self.is_in_end_game = True
        # Initialize Q-table with all zeros
        self.q_table_reward = self.generate_q_table_rewards(obs)
        self.q_table = self.generate_q_table(obs)
        

    def generate_q_table(self, observation: np.ndarray):
        """
        Generates a Q-table with all zeros.

        Args:
            shape: The shape of the Q-table.
        """
        #initialize q_table with all zeros
        return np.zeros((len(self.actions),observation.shape[0],observation.shape[1])).astype(float)
    
    # generate_q_table_rewards
    def generate_q_table_rewards(self, observation: np.ndarray):
        """
        Generates a Q-table with all zeros.
        And with rewards for obstacles, dirt, and home

        Args:
            shape: The shape of the Q-table.
        """
        #initialize q_table with all zeros
        q_table = np.zeros(observation.shape).astype(float)

        #set q_table values for obstacles, dirt, and home
        for i in range(observation.shape[0]):
            for j in range(observation.shape[1]):
                #set q_table values for obstacles, dirt, and home
                if observation[i][j] == 2 or observation[i][j] == 1:
                    q_table[i][j] = -3
                elif observation[i][j] == 3:
                    q_table[i][j] = 3
                elif observation[i][j] == 4:
                    q_table[i][j] = 10
                
        #   
        return q_table
    
    def compute_reward(self, x: int, y:int, dirt: np.ndarray, obstacles: np.ndarray) -> float:
        if x < 0 or x >= self.q_table.shape[1] or y < 0 or y >= self.q_table.shape[2]:
            # If the agent wants to hits a wall, add a penalty of -1
            print("Wall hit",-self.q_table_reward[x][y])
            return -self.q_table_reward[x][y]
        elif (x, y) in dirt:
            # If the agent collects a coin, add a reward of +1
            dirt.remove((x, y))
            #print("Dirt collected",self.q_table_reward[x][y])
            return self.q_table_reward[x][y]
        elif (x, y) in obstacles:
            # If the agent hits an obstacle, add a penalty of -1
            print("Obstacle hit",-self.q_table_reward[x][y])
            return -self.q_table_reward[x][y]
        else:
            # Use a heuristic to encourage the agent to move towards coins and avoid obstacles
            dirt_distances = [np.sqrt((x-c[0])**2 + (y-c[1])**2) for c in dirt]
            obstacle_distances = [-np.sqrt((x-o[0])**2 + (y-o[1])**2) for o in obstacles]
            final_distances = dirt_distances + obstacle_distances
            # Compute the heuristic-based reward
            if len(final_distances) > 0:
                #print("Reward: ", (-1) * 100 * np.min(final_distances) / np.max(final_distances))
                return (-1) * 0.1 * np.min(final_distances) / np.max(final_distances)
            else:
                return 0


    def process_reward(self, observation: np.ndarray, reward: float):
            """Any code that processes a reward given the observation is here.

            Args:
                observation: The observation which is returned by the environment.
                reward: The float value which is returned by the environment as a
                    reward.
            """

            #print("Reward: ", reward)
            #self.reward = reward
            pass

    

    def take_action(self, observation: np.ndarray, info: None | dict) -> float:
        """Any code that does the action should be included here.

        Args:
            observation: The observation which is returned by the environment.
            info: Any additional information your agent needs can be passed
                in here as well as a dictionary.
        """
        #get x and y coordinates of agent
        x, y = info["agent_pos"][self.agent_number]
        #print(observation)
        if self.show_info:
            self.print_state_info()
        #Q-Learning
        self.QLearning(x, y, observation,self.home, 200)
        #print(self.q_table)
        return self.get_next_best_move(x,y)
        
    
    def QLearning(self, x: int, y:int, observation: np.ndarray, final_state: tuple, max_steps_per_episode:int) -> None:
        #initialize next_x and next_y
        next_x,next_y = x,y
        # Train the agent default for 100 episodes
        for episode in range(self.train_range):
            steps = 0
            dirts = self.dirts.copy()
            while steps < max_steps_per_episode and (x,y) != final_state:
                # Get the next action
                action = self.get_next_move(x, y)
                #Action boundry check
                # Move the agent and get the reward
                if action == self.actions["UP"]:
                    next_x, next_y = x, y-1
                elif action == self.actions["DOWN"]:
                    next_x, next_y = x, y+1
                elif action == self.actions["LEFT"]:
                    next_x, next_y = x-1, y
                elif action == self.actions["RIGHT"]:
                    next_x, next_y = x+1, y      

                # Check if the agent is in a valid position
                if 1 <= next_x <= self.q_table.shape[1]-1 and 1 <= next_y <= self.q_table.shape[2]-1 and (next_x, next_y) not in self.obstacles:
                    if len(dirts)==0 and self.is_in_end_game == True:
                        dirts.append((final_state[0][0],final_state[1][0]))
                        print("Added Home to dirts")
                        self.is_in_end_game = False
                    # Get the reward
                    reward = self.compute_reward(x, y,dirts,self.obstacles)
                    # Update the Q-table
                    self.update_Q(x, y, action, reward, next_x, next_y)
                    # Update the position of the agent
                    x, y = next_x, next_y
                
                # Decay epsilon after each episode
                self.epsilon = 1.0 / ((episode / 50) + 1)
                # Check if the episode is done
                steps += 1

    def update_Q(self, x: int, y:int, action_idx:int, reward: int, next_x: int, next_y: int) -> None:
        # Get the maximum Q-value for the next state
        next_max = np.max(self.q_table[:, next_x, next_y])
        # Calculate the new Q-value
        current_value = self.q_table[action_idx][x][y]
        # Update the Q-value
        new_value = (1 - self.alpha) * current_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[action_idx][x][y] = new_value
        
        
    def get_next_move(self, x: int, y:int) -> float:
        if np.random.uniform(0, 1) < self.epsilon:
            # Choose a random action with probability epsilon
            #print("Random action")
            action = self.actions[np.random.choice(["UP","DOWN","RIGHT","LEFT"])]
        else:
            # Choose the best action with probability 1 - epsilon
            #print("Best action")
            action = np.argmax(self.q_table[:,x,y])
        return action

    def get_next_best_move(self, x: int, y:int) -> float:
        # Choose a random action with probability epsilon
        action_idx = np.argmax(self.q_table[:,x,y])
        #print(action_idx)
        for k,v in self.actions.items():
            if v == action_idx:
                f = open("./results/path.txt", "a")
                #print("Action: ", k, "Action Index: ", action_idx, "Q-Table: ", self.q_table[:,x,y], "X: ", x, "Y: ", y)
                f.write("Action: " + str(k) + " Action Index: " + str(action_idx) + " Q-Table: " + str(self.q_table[:,x,y]) + " X: " + str(x) + " Y: " + str(y))
                f.write("\n")
                f.close()
        return action_idx

    def print_state_info(self) -> None:
        print(self.q_table)
        print(self.observation)
        print("Current reward value:" ,self.reward)
        print("Current alpha value:" ,self.alpha)
        print("Current epsilon value:" ,self.epsilon)
        print("Current gamma value:" ,self.gamma)