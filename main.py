import pygame
import sys
import numpy as np
import pymunk
import pymunk.pygame_util
import math
from car_ import Car
pygame.init()
space = pymunk.Space()
space.gravity = (0, 0)
screen = pygame.display.set_mode((1670, 1000))
draw_options = pymunk.pygame_util.DrawOptions(screen)
clock = pygame.time.Clock()
pygame.display.set_caption("Гонки на Python")
background = pygame.image.load('road.png')
font = pygame.font.Font(None, 36)

car = Car(400, 300, space)

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
    
    # Отрисовка
    screen.blit(background, (0, 0))
    car.draw(screen)
    car.draw_steering_info(screen, font)  # Дебаг информация
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()