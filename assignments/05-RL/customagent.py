import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, action, reward, next_state, done = zip(*[self.buffer[i] for i in batch])
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class Agent:
    """
    A custom agent for the environment.
    """

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dqn = DQN(observation_space.shape[0], action_space.n).to(self.device)
        self.target_dqn = DQN(observation_space.shape[0], action_space.n).to(
            self.device
        )
        self.optimizer = optim.Adam(self.dqn.parameters())
        self.replay_buffer = ReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.target_update_frequency = 1000
        self.learning_steps = 0

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Return an action given an observation.
        """
        with torch.no_grad():
            state = (
                torch.tensor(observation, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            q_values = self.dqn(state)
        return int(q_values.argmax().item())

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Learn from a single timestep.
        """
        self.replay_buffer.push(observation, reward, terminated, truncated)
        if len(self.replay_buffer) < self.batch_size:
            return

        self.learning_steps += 1
        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size
        )
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.bool).to(self.device)

        q_values = self.dqn(state)
        next_q_values = self.target_dqn(next_state)
        target_q_values = q_values.clone()
        target_q_values[
            range(self.batch_size), action
        ] = reward + self.gamma * next_q_values.max(1)[0] * (~done)

        loss = nn.functional.mse_loss(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.learning_steps % self.target_update_frequency == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())
