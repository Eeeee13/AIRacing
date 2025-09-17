import pygame
import sys
import numpy as np
import pymunk
import pymunk.pygame_util
import math

pygame.init()
space = pymunk.Space()
space.gravity = (0, 0)
screen = pygame.display.set_mode((1670, 1000))
draw_options = pymunk.pygame_util.DrawOptions(screen)
clock = pygame.time.Clock()
pygame.display.set_caption("Гонки на Python")
background = pygame.image.load('road.png')


class Car:
    def __init__(self, x, y):
        self.original_image = pygame.image.load('red_car.png')
        self.image = pygame.image.load('red_car.png')
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)
        self.angle = 0
        self.rotation_speed = 1

        # Pymunk часть
        mass = 1000
        size = (self.rect.width * 0.8, self.rect.height * 0.6)  # Чуть меньше визуального размера
        moment = pymunk.moment_for_box(mass, size)
        
        self.body = pymunk.Body(mass, moment)
        self.body.position = x, y
        self.body.angle = math.radians(self.angle)
        
        self.shape = pymunk.Poly.create_box(self.body, size)
        self.shape.elasticity = 0.2
        self.shape.friction = 0.9
        self.shape.collision_type = 1  # Для обработки столкновений
        
        space.add(self.body, self.shape)

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    def update_graphics(self):
        """Синхронизация графики с физикой"""
        self.rect.center = (int(self.body.position.x), int(self.body.position.y))
        self.angle = math.degrees(self.body.angle)
        self.image = pygame.transform.rotate(self.original_image, -self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)
    
    def apply_engine_force(self, force):
        """Применение силы двигателя вперед"""
        # Сила применяется в локальных координатах машины
        self.body.apply_force_at_local_point((0, -force), (0, 0))
    
    def apply_brake(self, force):
        """Торможение"""
        self.body.apply_force_at_local_point((0, force), (0, 0))

    # def rotate(self):
    #     self.image = pygame.transform.rotate(self.original_image, -self.angle)
    #     old_center = self.rect.center
    #     self.rect = self.image.get_rect()
    #     self.rect.center = old_center  

    # def move(self, force):
    #     self.rotate()
    #     self.accel = 1
    #     speed = self.speed + self.accel
        
        

running = True
car = Car(400, 500)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        car.body.angular_velocity = -1
    if keys[pygame.K_RIGHT]:
        car.body.angular_velocity = 1
    if keys[pygame.K_UP]:
        car.apply_engine_force(50000)
    if keys[pygame.K_DOWN]:
        car.apply_brake(30000)


    # Обновление физики
    space.step(1/60)
    
    # Синхронизация графики
    car.update_graphics()
    
    # Отрисовка
    screen.fill((0, 0, 0))
    space.debug_draw(draw_options)  # Отрисовка физических объектов
    screen.blit(car.image, car.rect)  # Отрисовка машины
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()