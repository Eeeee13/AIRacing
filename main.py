import pygame
import sys
import numpy as np
import pymunk
import pymunk.pygame_util
import math
from car_ import Car
from enviroment import Track

pygame.init()
space = pymunk.Space()
space.gravity = (0, 0)
screen = pygame.display.set_mode((1670, 1000))
draw_options = pymunk.pygame_util.DrawOptions(screen)
clock = pygame.time.Clock()
pygame.display.set_caption("Гонки на Python")
background = pygame.image.load('road.png')
font = pygame.font.Font(None, 36)
track = Track('road.png')
track.create_mask_from_color(background)
# track_mask = track.create_mask_from_color(background)

car = Car(400, 300, space)
total_reward = 0
episode_reward = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    
    keys = pygame.key.get_pressed()
    
    # Обновление машины
    car.update(keys)
    
    # Обновление физики
    space.step(1/60)

    car.cast_rays(track)

    if track.check_collision(car.mask, (car.rect.x, car.rect.y)):
        reward = car.get_reward(crashed=True)  # Штраф за столкновение
        episode_reward += reward
        car.reset_to_start()
        print(f"Столкновение! Награда за эпизод: {episode_reward}")
        episode_reward = 0  # Сброс награды для нового эпизода
    else:
        reward = car.get_reward(crashed=False)  # Обычная награда
        episode_reward += reward

   
    # Отрисовка
    
    screen.blit(background, (0, 0))
    car.draw(screen)
    car.draw_steering_info(screen, font)  # Дебаг информация

    car.draw_rays(screen, track)
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()