import pygame
import torch
import numpy as np
from environment import Environment
from ppo_agent import PPOPolicy, select_action, ppo_update
from config import *

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

env = Environment()
policy = PPOPolicy(STATE_DIM, ACTION_DIM)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

NUM_EPISODES = 1000
GAMMA = 0.99

for episode in range(NUM_EPISODES):
    state = env.reset()
    # print("state.shape =", len(state), "example:", state)

    log_probs, values, rewards, states, actions = [], [], [], [], []
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # print("AA State.shape =", len(state), "example:", state)
        action, log_prob, value = select_action(policy, state)
        next_state, reward, done = env.step(action)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        states.append(state)
        actions.append(action)

        state = next_state

        env.render(screen)
        pygame.display.flip()
        clock.tick(FPS)

    # подготовка батча
    returns, adv = [], []
    G, A = 0, 0
    for r, v in zip(reversed(rewards), reversed(values)):
        G = r + GAMMA * G
        A = G - v.item()
        returns.insert(0, G)
        adv.insert(0, A)

    states = torch.FloatTensor(np.array(states))
    actions = torch.FloatTensor(np.array(actions))
    log_probs_old = torch.stack(log_probs)
    returns = torch.FloatTensor(returns)
    advantages = torch.FloatTensor(adv)

    ppo_update(policy, optimizer, states, actions, log_probs_old, returns, advantages)

    print(f"Episode {episode}, total reward {sum(rewards)}")
