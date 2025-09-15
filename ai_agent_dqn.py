# ai_agent_dqn.py
import random
import collections
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import (ACTION_SPACE, DQN_BATCH_SIZE, DQN_GAMMA, DQN_LR,
                    DQN_EPS_START, DQN_EPS_END, DQN_EPS_DECAY,
                    DQN_TARGET_UPDATE, DQN_REPLAY_SIZE, DQN_MIN_REPLAY,
                    DQN_DEVICE, SEED)

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# простой feed-forward net: вход — состояние (сенсоры + угол + скорость при необходимости)
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(128, 128)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# replay buffer
Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, device=None):
        self.device = device or DQN_DEVICE
        self.state_dim = state_dim
        self.action_dim = ACTION_SPACE

        self.policy_net = QNetwork(state_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=DQN_LR)
        self.replay = ReplayBuffer(DQN_REPLAY_SIZE)

        self.steps_done = 0
        self.eps_start = DQN_EPS_START
        self.eps_end = DQN_EPS_END
        self.eps_decay = DQN_EPS_DECAY
        self.gamma = DQN_GAMMA
        self.batch_size = DQN_BATCH_SIZE
        self.min_replay = DQN_MIN_REPLAY
        self.target_update = DQN_TARGET_UPDATE

    def select_action(self, state, evaluate=False):
        """state: numpy array (state_dim,)"""
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if evaluate or random.random() > eps_threshold:
            # greedy
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                qvals = self.policy_net(s)
                action_idx = qvals.argmax(dim=1).item()
                return action_idx  # индекс в ACTIONS
        else:
            return random.randrange(self.action_dim)

    def store_transition(self, state, action_idx, reward, next_state, done):
        self.replay.push(
            np.array(state, dtype=np.float32),
            int(action_idx),
            float(reward),
            np.array(next_state, dtype=np.float32) if next_state is not None else None,
            bool(done)
        )

    def optimize(self):
        if len(self.replay) < max(self.batch_size, self.min_replay):
            return None  # ещё не готовы

        transitions = self.replay.sample(self.batch_size)
        # convert to tensors
        state_batch = torch.tensor(np.vstack(transitions.state), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(transitions.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(transitions.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        non_final_mask = torch.tensor([s is not None for s in transitions.next_state], dtype=torch.bool, device=self.device)
        non_final_next_states = torch.tensor(
            np.vstack([s for s in transitions.next_state if s is not None]),
            dtype=torch.float32, device=self.device
        ) if any(non_final_mask) else None

        # Q(s, a)
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # compute target
        with torch.no_grad():
            next_q_values = torch.zeros(self.batch_size, 1, device=self.device)
            if non_final_next_states is not None:
                # max_a' Q_target(next_state, a')
                max_next_q = self.target_net(non_final_next_states).max(1)[0].unsqueeze(1)
                next_q_values[non_final_mask] = max_next_q
            expected_q = reward_batch + (self.gamma * next_q_values * (~torch.tensor(transitions.done, device=self.device).unsqueeze(1)))

        # MSE loss
        loss = nn.functional.mse_loss(q_values, expected_q)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping optional
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # periodically update target net
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path_prefix="dqn_agent"):
        torch.save({
            'policy_state': self.policy_net.state_dict(),
            'target_state': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'replay_buffer': list(self.replay.buffer)
        }, f"{path_prefix}.pth")

    def load(self, path_prefix="dqn_agent"):
        data = torch.load(f"{path_prefix}.pth", map_location=self.device)
        self.policy_net.load_state_dict(data['policy_state'])
        self.target_net.load_state_dict(data['target_state'])
        self.optimizer.load_state_dict(data['optimizer'])
        self.steps_done = data.get('steps_done', 0)
        # replay buffer restore (simple)
        buf = data.get('replay_buffer', [])
        self.replay.buffer = collections.deque(buf, maxlen=self.replay.capacity)
