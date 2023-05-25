import numpy as np
from collections import deque
from agents import BaseAgent
from copy import deepcopy


class MCAgent(BaseAgent):
    def __init__(
        self,
        agent_number,
        obs: np.ndarray,
        gamma,
        epsilon,
        len_episode,
        n_times_no_policy_change_for_convergence,
    ):
        """
        Initialize the agent.

        Parameters
        ----------
        agent_number : int
            The number of the agent. This is used to identify the agent in the
            environment.
        obs : np.ndarray
            The initial observation of the environment.
        gamma : float
            The discount factor.
        epsilon : float
            The probability of choosing a random action instead of the optimal
            one.
        len_episode : int
            The maximum length of an episode.
        n_times_no_policy_change_for_convergence : int
            The number of times the optimal policy has to be the same before
            the agent considers it converged.
        """
        super().__init__(agent_number)

        self.max_len_episode = len_episode
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_times_no_policy_change_for_convergence = (
            n_times_no_policy_change_for_convergence
        )
        self.times_finished = 0

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

        # Create a 3D array named Q, where each entry is a dictionary
        # with keys the actions and values the Q values
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

        # Create a 2D array named policy, where each entry is an integer
        # representing the action to take in each state
        self.policy = np.zeros((self.x_size, self.y_size))
        self.old_optimal_policy = np.zeros((self.x_size, self.y_size))
        self.optimal_policy = np.zeros((self.x_size, self.y_size))
        self.constant_optimal_policy_counter = 0

        # create initial policy
        self.update_policy()

        # to keep track of (s,a,r) of each step in episode
        # we use a deque for efficiency
        self.episode = deque()
        self.episode_rewards = deque()

    def process_reward(self, obs: np.ndarray, reward: float, info):
        """
        Check if terminated (we assume that happens if charging station reached) Or maximum number of steps reached in episode.
        If so -> update Q and policy and reset episode.

        Parameters
        ----------
        obs : np.ndarray
            The observation of the environment.
        reward : float
            The reward obtained from the environment.
        info : dict
            Additional information from the environment.
        """

        # Add reward obtained to the list with rewards
        self.episode_rewards.append(reward)

        # Check if terminated (=end of episode or charging station reached with all dirt cleaned)
        if reward == 10 or len(self.episode_rewards) >= self.max_len_episode:
            # if the reward is 10, the charging station was reached with all dirt cleaned
            # so times_finished is increased by 1
            if reward == 10:
                self.times_finished += 1

            # update Q and policy
            self.update_Q()
            self.update_policy()

            # reset episode for next iteration
            self.reset_episode()

            return self.check_convergence(reward == 10)

        return False  # in order to always return something, not only after checking convergence

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
            [[False for _ in range(len(self.A))] for _ in range(self.y_size)]
            for _ in range(self.x_size)
        ]

        # create deque that keeps track of the state action pairs seen in the episode
        state_action_pairs_seen = deque()

        for index, ((currentX, currentY), currentA) in enumerate(self.episode):
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

    def check_convergence(self, last_route_succesful):
        """
        Check if the policy has converged. This is done by comparing the current optimal policy to the
        previous optimal policy. Also, we check if this function is called after succesfully reaching
        the charging station. This is needed in order to allow convergence. If the agent has not reached
        the charging station, the policy might have stayed the same because the agent has not learned the
        right policy yet, but did not find an improvement either.

        Parameters
        ----------
        last_route_succesful : bool
            Checks if this function is called after succesfully reaching the charging station.

        """
        # only check for convergence after the end is reached a number of times
        # this prevents the agent from converging too early, as it might not have
        # seen all states yet
        if self.times_finished > 4:
            if np.array_equal(self.optimal_policy, self.old_optimal_policy):
                self.constant_optimal_policy_counter += 1
            else:
                self.constant_optimal_policy_counter = 0

            output = (
                True
                if self.constant_optimal_policy_counter
                >= self.n_times_no_policy_change_for_convergence
                and last_route_succesful
                else False
            )
            return output

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        """
        Return the action that should be taken by the agent, based on the current
        policy and state (= (x,y)).

        Args:
            observation (np.ndarray): The current observation of the environment.
            info (None | dict): The info dict of the environment.

        """
        # Get the x,y values of the state
        x, y = info["agent_pos"][self.agent_number]
        x, y = int(x), int(y)

        # For current state, get the action based on the current policy
        next_action = self.policy[x][y]

        # Log the state action pair in self.episode
        self.episode.append(((x, y), next_action))

        # return the action we take
        return next_action