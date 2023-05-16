"""Greedy Agent.

Chooses the best scoring value with no thought about the future.
"""
import numpy as np
from random import randint

from collections import deque
from agents import BaseAgent


class MCAgent(BaseAgent):
    def __init__(self, agent_number, obs: np.ndarray):
        """Chooses an action randomly unless there is something neighboring.

        Args:
            agent_number: The index of the agent in the environment.
        """
        super().__init__(agent_number)
        self.dirts = [(x, y) for x, y in zip(np.where(obs == 3)[0],np.where(obs == 3)[1])]
        print("after initialization, self. dirst = {self.dirts}")
        self.dirt_found = [0 for _ in range(len(self.dirts))]
        
        self.maxD = 2**len(self.dirts)

        self.x_size, self.y_size = obs.shape
        self.A = [0, 1, 2, 3, 4]

        # Create a 3D array named Returns, where each entry is a dictionary
        # the dictionary contains the following keys:
        # "total" = the total score of the episode
        # "n" = the number of times the state was visited
        self.Returns = [
            [
                [
                    [
                    {"total": 0, "n": 0} for _ in range(len(self.A)) ]
                    for _ in range(self.maxD) ]
                for _ in range(self.y_size) ]
            for _ in range(self.x_size) ]
        #print(f"self.returns is {self.Returns}")

        self.Q = [
            [   
                [
                    {
                        0: np.random.rand(),
                        1: np.random.rand(),
                        2: np.random.rand(),
                        3: np.random.rand(),
                        4: np.random.rand(),
                    }
                    for _ in range(self.maxD) ]
                for _ in range(self.y_size) ]
            for _ in range(self.x_size) ]
        
        self.policy = np.zeros((self.x_size, self.y_size, self.maxD))
        # create initial policy
        self.generate_initial_policy()

        # to keep track of (s,a,r) of each step in episode
        self.episode = deque()
        self.episode_rewards = deque()

        # Initialize episode counter and charging status
        self.charging = False

        # Variables to set (maybe as parameters in the constructor)
        self.max_len_episode = 50
        self.gamma = 0.5
        self.epsilon = 0.1

    def list_to_int(self, d_list):
        #Convert a list of zeroes and ones to a binary number
        #e.g. [0,1,0,1] -> 5
        d_list = [str(i) for i in d_list]
        d_string = "".join(d_list)
        d_int = int(d_string, 2)
        return d_int
    
    def generate_initial_policy(self):
        # loop over all x,y in self.Q
        for x in range(self.x_size):
            for y in range(self.y_size):
                for d in range(self.maxD):
                    # get the actions for this state
                    actions = self.Q[x][y][d]
                    # find the key that corresponds to the max value
                    max_key = max(actions, key=actions.get)
                    # set the policy at x,y to the max key
                    self.policy[x, y, d] = max_key

    def process_reward(self, obs: np.ndarray, reward: float, info):
        """
        Check if terminated (we assume that happens if charging station reached) Or maximum number of steps reached in episode.
        If so -> update Q and policy

        """
        self.episode_rewards.append(reward)
        #print(f"reward is:{reward} ")
        if reward == 10 or len(self.episode) >= self.max_len_episode:
            print("updating policy")
            # update Q and policy
            self.update_Q()
            self.update_policy()
            # reset episode
            self.episode = deque()
            self.episode_rewards = deque()
            pass
        
        elif reward == 5:
            #find the index of the dirt that was cleaned
            x, y = info["agent_pos"][self.agent_number]



#Zo nog
    def update_Q(self):
        # Create a new 3D array named newQ to keep track of the state action pairs seen in the episode
        # initially all entries are False. As soon as we see a state action pair, we set the corresponding
        # cell to a dict which keeps track of the discounted reward
        newQ = [
                [
                    [
                        [False for _ in range(len(self.A))] 
                        for _ in range(self.maxD) ]
                    for _ in range(self.y_size)]
                for _ in range(self.x_size)]
        
        state_action_pairs_seen = deque()
        print("first five entries of the epsiode are", self.episode)
        for index, ((currentX, currentY, currentD), currentA) in enumerate(self.episode):
            #print(f"currentX is {currentX}, currentY is {currentY}, currentD is {currentD}, currentA is {currentA}")
            currentX, currentY, currentD, currentA = int(currentX), int(currentY), int(currentD), int(currentA)
            # This loop is designed in such a way that we have to loop over
            # the state action pairs seen in the episode only once. As we loop

            # over them, we update the discounted rewards for each state action pair.
            # we do this by keeping a dict for each seen state action pair, which contains
            # the current discounted reward value and the current gamma exponent value.
            currentReward = self.episode_rewards[index]

            if index < 6:
                print(f"currentX is {currentX}, currentY is {currentY}, currentD is {currentD}, currentA is {currentA}")
                print(f"currentReward is {currentReward}")

            # Update all discounted rewards for previously spotted state action pairs
            for prevX, prevY, prevD, prevA in state_action_pairs_seen:
                discounted_reward_dict = newQ[prevX][prevY][prevD][prevA]

                new_gamma_exponent = discounted_reward_dict["gamma_exponent"] + 1
                discounted_reward_dict["gamma_exponent"] = new_gamma_exponent

                discounted_reward_dict["discounted_reward"] += (
                    self.gamma**new_gamma_exponent * currentReward
                )

            # Create dict for this state action pair if is the first time we are seeing it
            # in this episode
            if not newQ[currentX][currentY][currentD][currentA]:  # means that this is the first time we are seeing this state action pair in this episode
                newQ[currentX][currentY][currentD][currentA] = {
                    "discounted_reward": currentReward * self.gamma,
                    "gamma_exponent": 1,
                }
                state_action_pairs_seen.append((currentX, currentY, currentD, currentA))

        # update Returns: total, n
        # and update Q[x][y][a] with the new AVG(!) discounted reward
        for x, y, d, a in state_action_pairs_seen:
            self.Returns[x][y][d][a]["total"] += newQ[x][y][d][a]["discounted_reward"]
            self.Returns[x][y][d][a]["n"] += 1



            self.Q[x][y][d][a] = (
                self.Returns[x][y][d][a]["total"] / self.Returns[x][y][d][a]["n"]
            )
        
        print(
            f"""
            self.policy is {self.policy[7][5][0]}
            self.Q is {self.Q[7][5][0]}
            newQ is {newQ[7][5][0]}
            self.Returns is {self.Returns[7][5][0]}
            """

        )

    def update_policy(self):
        # loop over all x,y in self.Q
        for x in range(self.x_size):
            for y in range(self.y_size):
                for d in range(self.maxD):
                    actions = self.Q[x][y][d]
                    # find the key that corresponds to the max value
                    max_key = max(actions, key=actions.get)

                    actions = 5 * [self.epsilon / len(self.A)]
                    actions[max_key] = 1 - self.epsilon + self.epsilon / len(self.A)

                    action = np.random.choice(a=5, p=actions)

                    # set the policy at x,y to the max key
                    self.policy[x, y, d] = action

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        """
        Take a step based on the current policy, and log it in self.episode (s,a,r).

        """
        x, y = info["agent_pos"][self.agent_number]
        x, y = int(x), int(y)

        print(f"self dirst is {self.dirts} ")
        if (x,y) in self.dirts:
            index = self.dirts.index((x, y))
            #set the corresponding entry in self.dirt_found to 1
            self.dirt_found[index] = 1
        
        d = self.list_to_int(self.dirt_found)


        self.charging = info["agent_charging"][self.agent_number]
        self.episode.append(((x, y, d), self.policy[x][y][d]))

        # take action based on current policy
        return self.policy[x][y][d]
