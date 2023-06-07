import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Deep Q Network (DQN) architecture
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Deep Double Q-learning agent
class DDQNAgent:
    def __init__(
        self,
        agent_number,
        state_size,
        action_size,
        hidden_size,
        learning_rate,
        gamma,
        epsilon,
    ):
        self.agent_number = agent_number
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.005
        self.max_epsilon = 1
        self.min_epsilon = 0.001
        self.tau = 0.05

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("the test will be run on: ", self.device)


        # Q-networks
        self.q_network = DQN(state_size, action_size, hidden_size).to(self.device)
        self.target_network = DQN(state_size, action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay memory
        self.memory = []

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        agent_pos = info["agent_pos"][self.agent_number]
        # state = observation

        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            #combine the agent position with the state into one torch tensor
            state = np.concatenate((observation.flatten(), agent_pos))
            state = torch.tensor(state, dtype=torch.float32).flatten().to(self.device)


            with torch.no_grad():
                q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def process_reward(
        self,
        observation: np.ndarray,
        info: None | dict,
        reward: float,
        old_state: tuple,
        next_state: tuple,
        action: int,
        done: bool,
    ):
        #agent_pos = info["agent_pos"][self.agent_number]
        #print(f"old_state looks like this: {old_state}")
        self.memory.append((old_state, action, reward, next_state, done))

        if self.epsilon < 0.0011:
            return True
        else:
            return False

    def replay(self, batch_size):
        #print(f"self.memory looks like this: {self.memory}")
        if len(self.memory) < batch_size:
            return

        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        #print(f"indices looks like this: {indices}")
        #print(f"self.memory looks like this: {self.memory[1][1]}")

        #Below one ran, but gave some issues before. Maybe switch back to this one if the other one doesn't work	
        #batch = np.array(self.memory)[indices]
        batch = [self.memory[i] for i in indices]

        #print(f"batch looks like this: {batch}")    
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = (
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        )
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # print(states.shape, actions.shape, rewards.shape, next_states.shape)

        q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).detach()
        max_next_actions = torch.argmax(
            self.q_network(next_states), dim=1, keepdim=True
        )

        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values.gather(
            1, max_next_actions
        )

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        target_net_state_dict   = self.target_network.state_dict()
        q_net_state_dict        = self.q_network.state_dict()
        for key in target_net_state_dict:
            target_net_state_dict[key] = (1 - self.tau) * target_net_state_dict[key] + self.tau * q_net_state_dict[key]
        self.target_network.load_state_dict(target_net_state_dict)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self, episode):
        
        self.epsilon = self.max_epsilon * (1- episode /100000)
        #keep a constant epsilon of 0.2
        #self.epsilon = 0.2


        # self.epsilon *= decay_rate
        #print(self.epsilon)
        # self.epsilon = self.min_epsilon + (
        #     self.max_epsilon - self.min_epsilon
        # ) * np.exp(-self.epsilon_decay * episode)

    def save_model(self, path):
        print("model_saved")
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.q_network.eval()
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
