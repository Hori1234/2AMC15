import numpy as np
from collections import deque
from agents import BaseAgent
from copy import deepcopy



class MCAgent(BaseAgent):
    def __init__(self, agent_number, obs: np.ndarray, gamma, epsilon, len_episode, n_times_no_policy_change_for_convergence, replace_agent_after_episode, replace_to_start):
        """TODO write docstring here"""
        super().__init__(agent_number)

        # Variables to set (maybe as parameters in the constructor)
        self.max_len_episode = len_episode
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_times_no_policy_change_for_convergence = n_times_no_policy_change_for_convergence
        self.replace_agent_after_episode = replace_agent_after_episode
        self.replace_to_start = replace_to_start #If false, place back at last cleaned dirt

        # Keeps track of our initial position, and last places we found dirt.
        # If we have bene moving for too long without success, we'll reset to
        # this location. Initial value will be set in teke_action because we
        # don't know our initial position yet.
        # self.reset_location = None

        self.x_size, self.y_size = obs.shape
        self.A = [0, 1, 2, 3, 4]

        # Create a 3D array named Returns, where each entry is a dictionary
        # the dictionary contains the following keys:
        # "total" = the total score of the episode
        # "n" = the number of times the state was visited
        self.Returns = [
            [
                
                [{"total": 0, "n": 0} for _ in range(len(self.A))]

                for _ in range(self.y_size)
            ]
            for _ in range(self.x_size)
        ]

        self.Q = [
            [
                {
                    0: np.random.rand(),
                    1: np.random.rand(),
                    2: np.random.rand(),
                    3: np.random.rand(),
                    4: np.random.rand(),
                }
                for _ in range(self.y_size)
            ]
            for _ in range(self.x_size)
        ]

        self.policy = np.zeros((self.x_size, self.y_size))
        self.old_optimal_policy = np.zeros((self.x_size, self.y_size))
        self.optimal_policy = np.zeros((self.x_size, self.y_size))
        self.constant_optimal_policy_counter = 0

        # create initial policy
        self.update_policy()

        # to keep track of (s,a,r) of each step in episode
        self.episode = deque()
        self.episode_rewards = deque()


    def process_reward(self, obs: np.ndarray, reward: float, info):
        """
        Check if terminated (we assume that happens if charging station reached) Or maximum number of steps reached in episode.
        If so -> update Q and policy and reset episode.
        """
        # Add reward obtained to the list with rewards
        self.episode_rewards.append(reward)

        # Check if terminated (=end of episode or charging station reached with all dirt cleaned)
        if reward == 10 or len(self.episode_rewards) >= self.max_len_episode:
            # update Q and policy
            self.update_Q()
            self.update_policy()

            # reset episode
            self.reset_episode()

            return self.check_convergence()
        
        return False #in order to always return something, not only after checking convergence

    def reset_episode(self):
        """
        Reset the deques that keep track of the episode state, action pairs and the episode rewards.
        """
        self.episode = deque()
        self.episode_rewards = deque()

    def update_Q(self):
        """
        After an episode is finished, or we've reached the charging station, update the Q values for all
        state action pairs seen in the episode.
        """
        # Create a new 3D array named newQ to keep track of the state action pairs seen in the episode
        # initially all entries are False. As soon as we see a state action pair, we set the corresponding
        # cell to a dict which keeps track of the discounted reward
        newQ = [
            [
                [False for _ in range(len(self.A))]
                for _ in range(self.y_size)
            ]
            for _ in range(self.x_size)
        ]

        # create deque that keeps track of the state action pairs seen in the episode
        state_action_pairs_seen = deque()

        for index, ((currentX, currentY), currentA) in enumerate(
            self.episode
        ):
            currentX, currentY, currentA = (
                int(currentX),
                int(currentY),

                int(currentA),
            )
            # This loop is designed in such a way that we have to loop over
            # the state action pairs seen in the episode only once. As we loop
            # over them, we update the discounted rewards for each state action pair.
            # we do this by keeping a dict for each seen state action pair, which contains
            # the current discounted reward value and the current gamma exponent value.
            currentReward = self.episode_rewards[index]

            # Update all discounted rewards for previously spotted state action pairs
            for prevX, prevY, prevA in state_action_pairs_seen:
                discounted_reward_dict = newQ[prevX][prevY][prevA]

                new_gamma_exponent = discounted_reward_dict["gamma_exponent"] + 1
                discounted_reward_dict["gamma_exponent"] = new_gamma_exponent

                discounted_reward_dict["discounted_reward"] += (
                    self.gamma**new_gamma_exponent * currentReward
                )

            # Create dict for this state action pair if is the first time we are seeing it
            # in this episode
            if not newQ[currentX][currentY][
                currentA
            ]:  # means that this is the first time we are seeing this state action pair in this episode
                newQ[currentX][currentY][currentA] = {
                    "discounted_reward": currentReward * self.gamma,
                    "gamma_exponent": 1,
                }
                state_action_pairs_seen.append((currentX, currentY, currentA))

        # update Returns: total, n
        # and update Q[x][y][a] with the new AVG(!) discounted reward
        for x, y, a in state_action_pairs_seen:
            self.Returns[x][y][a]["total"] += newQ[x][y][a]["discounted_reward"]
            self.Returns[x][y][a]["n"] += 1

            self.Q[x][y][a] = (
                self.Returns[x][y][a]["total"] / self.Returns[x][y][a]["n"]
            )

    def update_policy(self, optimal=False):
        """
        Update the policy for each state value.

        This is done by looking at the Q values for each state and choosing between
        the action with the highest Q value with probability 1-epsilon + epsilon/|A|
        and a random different action with probability epsilon/|A|.

        Args:
            optimal (bool, optional): If True, we take the optimal policy. If false,
                we take the epsilon greedy policy. Defaults to False.
        """
        self.old_optimal_policy = deepcopy(self.optimal_policy)

        # loop over all x,y in self.Q
        for x in range(self.x_size):
            for y in range(self.y_size):
                Q_values = self.Q[x][y]
                # find the key that corresponds to the max value
                max_key = max(Q_values, key=Q_values.get)

                # If training is over, we want to take the optimal policy
                if optimal:
                    action = max_key
                else:
                    action_probs = len(self.A) * [self.epsilon / len(self.A)]

                    action_probs[max_key] = (
                        1 - self.epsilon + self.epsilon / len(self.A)
                    )

                    action = np.random.choice(a=5, p=action_probs)

                # set the policy at x,y,d to the chosen action
                self.policy[x, y] = action
                self.optimal_policy[x, y] = max_key
    
    def check_convergence(self):
        if np.array_equal(self.optimal_policy, self.old_optimal_policy):
            self.constant_optimal_policy_counter += 1
            # print(f"""
            # No change in policy.
            # counter: {self.constant_optimal_policy_counter}
            # """)            
        else:
            self.constant_optimal_policy_counter = 0
        
        return True if self.constant_optimal_policy_counter >= self.n_times_no_policy_change_for_convergence else False


    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        """
        Return the action that should be taken by the agent, based on the current
        policy and state (= (x,y)).
        """
        # If we are at the end of an episode, move back to the starting position 
        # or the last place dirt was found
        if (len(self.episode) == self.max_len_episode - 1) and self.replace_agent_after_episode:
            return 5 if self.replace_to_start else 6

        # Get the x,y values of the state
        x, y = info["agent_pos"][self.agent_number]
        x, y = int(x), int(y)

        # If this is the initial location, set this as place to reset to
        # self.reset_location = (x, y)


        # For current state, get the action based on the current policy
        next_action = self.policy[x][y]

        # Log the state action pair in self.episode
        self.episode.append(((x, y), next_action))

        # return the action we take
        return next_action

