from stable_baselines3 import PPO
from racing_env import make_racing_env
import pygame

# Создаём окружение напрямую
env = make_racing_env(render_mode="human")

# Загружаем модель (без передачи env внутрь)
model = PPO.load("racing_models/best_model.zip")

obs, _ = env.reset()

done = False
env.render
race = False
while not race:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                race = True
            

while race:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()


env.close()
