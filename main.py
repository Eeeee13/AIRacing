# main.py
import pygame
import numpy as np
from environment import Environment
from ai_agent_dqn import DQNAgent
from config import WIDTH, HEIGHT, FPS, ACTIONS, NUM_SENSORS

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

env = Environment()
# state_dim = NUM_SENSORS + 1 (угол) — нормализуем угол в [-1,1]
state_dim = NUM_SENSORS + 1
agent = DQNAgent(state_dim=state_dim)

running = True
episode_reward = 0
episode = 0

# вспомогательные ф-ии
def build_state(sensors, car_angle):
    # нормализуем расстояния в [0,1] по SENSOR_RANGE
    # допустим у тебя SENSOR_RANGE в config; если нет — нормализуй на max возможную длину
    sensors = np.array(sensors, dtype=np.float32)
    # если в sensors лежит SENSOR_RANGE как max — нормализуем 
    sensors = sensors / sensors.max() if sensors.max() > 0 else sensors
    # угол нормализуем в [-1,1] (угол в градусах / 180)
    angle_norm = np.array([car_angle / 180.0], dtype=np.float32)
    return np.concatenate([sensors, angle_norm])

# reset env
sensors = env.reset()
car_angle = env.car.angle
state = build_state(sensors, car_angle)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # выбор действия индексом
    action_idx = agent.select_action(state)
    action_value = ACTIONS[action_idx]

    # делаем шаг
    next_sensors, reward, done = env.step(action_value)
    next_state = build_state(next_sensors, env.car.angle) if not done else None

    # store and learn
    agent.store_transition(state, action_idx, reward, next_state, done)
    loss = agent.optimize()  # вернёт loss или None

    episode_reward += reward

    # подготовка next
    if done:
        episode += 1
        print(f"Episode {episode}, reward {episode_reward}, replay {len(agent.replay)}")
        episode_reward = 0
        sensors = env.reset()
        state = build_state(sensors, env.car.angle)
    else:
        state = next_state

    # render
    env.render(screen)
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
