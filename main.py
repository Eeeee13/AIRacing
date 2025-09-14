import pygame
from environment import Environment
from ai_agent import QLearningAgent
from config import WIDTH, HEIGHT, FPS

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

env = Environment()
agent = QLearningAgent()

running = True
state = env.reset()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = agent.choose_action(state)
    next_state, reward, done = env.step(action)
    agent.learn(state, action, reward, next_state)
    state = next_state if not done else env.reset()

    env.render(screen)
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
