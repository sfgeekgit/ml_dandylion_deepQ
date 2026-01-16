import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# Hyperparameters
STATE_SIZE = 8
ACTION_SIZE = 8
EPISODES = 1000
LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
BATCH_SIZE = 32
MEMORY_SIZE = 1000

HIDDEN_SIZE = 12

reward_vals = {
    "win": 1000, 
    "illegal": -100, 
    "meh": 10 
}

# Neural Network for Q-Learning
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Experience Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Deep Q-Learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        self.epsilon = EPSILON

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice([i for i in range(self.action_size) if state[i] == 0])
        state = torch.FloatTensor(state)
        with torch.no_grad():
            q_values = self.model(state)
        q_values = q_values * torch.FloatTensor([state[i] == 0 for i in range(self.action_size)])
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = self.memory.sample(BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state)
                target = reward + GAMMA * torch.max(self.target_model(next_state)).item()
            state = torch.FloatTensor(state)
            target_f = self.model(state)
            target_f = target_f.clone()
            target_f[action] = target
            output = self.model(state)
            loss = self.loss_fn(output, target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY

    def compute_reward(self, state, next_state, action):
        # Check if the action is illegal (repeated guess)
        if state[action] == 1:
            return reward_vals["illegal"]
        # Check if the agent has made 7 correct guesses without a repeat
        if sum(next_state) == 7:
            return reward_vals["win"]
        # Default case, return "meh" reward
        return reward_vals["meh"]

    def train(self, episodes):
        for e in range(episodes):
            state = np.zeros(STATE_SIZE)
            for time in range(STATE_SIZE):
                action = self.act(state)
                next_state = state.copy()
                next_state[action] = 1
                reward = self.compute_reward(state, next_state, action)
                done = sum(next_state) == 7
                self.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    self.update_target_model()
                    break
                self.replay()
            if e % 100 == 0:
                print(f"Episode: {e}/{episodes}, Epsilon: {self.epsilon:.2}")

# Initialize and train the agent
agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
agent.train(EPISODES)
