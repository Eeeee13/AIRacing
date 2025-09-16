import pygame
import math
import numpy as np
from config import WIDTH, HEIGHT, SENSOR_COUNT, MAX_SPEED, ACCELERATION, FRICTION, TURN_SPEED

class Car:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.angle = 0  # направление в градусах
        self.speed = 0
        self.max_speed = 6
        self.acceleration = 0.2
        self.friction = 0.05
        self.turn_speed = 4
        self.alive = True

    def reset(self, x=None, y=None, angle=0):
        self.x = x if x else WIDTH // 2
        self.y = y if y else HEIGHT // 2
        self.angle = angle
        self.speed = 0
        self.steer_angle = 0
        self.acc = 0
        self.alive = True

    def update(self, action):
        if not self.alive:
            return
        steer, throttle = action

        # обновляем ускорение
        self.acc = throttle * ACCELERATION
        self.speed += self.acc
        self.speed = max(-MAX_SPEED/2, min(self.speed, MAX_SPEED))

        # трение
        if self.speed > 0:
            self.speed -= FRICTION
            if self.speed < 0: self.speed = 0
        elif self.speed < 0:
            self.speed += FRICTION
            if self.speed > 0: self.speed = 0

        # поворот (чем быстрее, тем сильнее эффект)
        self.angle += steer * TURN_SPEED * (self.speed / MAX_SPEED)

        # движение
        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y += self.speed * math.sin(rad)



    def draw(self, screen):
        car_rect = pygame.Rect(0, 0, 20, 40)
        car_rect.center = (self.x, self.y)
        rotated = pygame.transform.rotate(pygame.Surface((20, 40)), -self.angle)
        rotated.fill((255, 0, 0))
        screen.blit(rotated, rotated.get_rect(center=car_rect.center))

    def get_sensors(self, walls):
        """8 лучей вокруг машины"""
        distances = []
        for i in range(SENSOR_COUNT):
            angle = math.radians(self.angle + (360 / SENSOR_COUNT) * i)
            dist = 0
            x, y = self.x, self.y
            while 0 < x < WIDTH and 0 < y < HEIGHT and dist < 200:
                x += math.cos(angle)
                y += math.sin(angle)
                dist += 1
            distances.append(dist / 200.0)  # нормируем
        return distances
    
    def get_state(self, walls):
        sensors = self.get_sensors(walls)   # список из 8 чисел
        norm_speed = self.speed / MAX_SPEED
        norm_acc = self.acc / ACCELERATION  # нормализуем ускорение
        norm_angle = math.sin(math.radians(self.angle))  # угол → [-1,1]
        state = np.array(sensors + [norm_speed, norm_acc, norm_angle], dtype=np.float32)
        return state

