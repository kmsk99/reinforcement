import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32)
                q_values = self.q_network(state)
                return np.argmax(q_values.numpy())
        else:
            return random.randrange(self.action_size)

    def train(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute Q(s, a)
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute Q(s', a) for all a
        next_q_values = self.target_network(next_states).max(1)[0]

        # Compute target value: r + γ max_a Q(s', a)
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = self.criterion(q_values, targets.detach())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update exploration rate
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # Optionally update target network
        # ...
