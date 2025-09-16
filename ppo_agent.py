import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.distributions as dist
from config import *

class PPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.mu_head = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.value_head = nn.Linear(128, 1)

    def forward(self, state):
        x = self.fc(state)
        mu = torch.tanh(self.mu_head(x))
        std = torch.exp(self.log_std)
        value = self.value_head(x)
        return mu, std, value

def select_action(policy, state):
    state = torch.FloatTensor(state).unsqueeze(0)
    print("States: ",state)
    mu, std, value = policy(state)
    dist_normal = dist.Normal(mu, std)
    action = dist_normal.sample()
    log_prob = dist_normal.log_prob(action).sum(dim=-1)
    print(action)
    return action.squeeze(0).detach().numpy(), log_prob.detach(), value.detach()

def ppo_update(policy, optimizer, states, actions, log_probs_old, returns, advantages, clip_eps=0.2):
    print("States: ",states)
    mu, std, values = policy(states)
    dist_normal = dist.Normal(mu, std)
    log_probs = dist_normal.log_prob(actions).sum(dim=-1)
    ratio = torch.exp(log_probs - log_probs_old)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = (returns - values.squeeze()).pow(2).mean()

    loss = policy_loss + 0.5 * value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
