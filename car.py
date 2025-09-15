import pygame, math
from config import CAR_SPEED, SENSOR_RANGE, NUM_SENSORS

class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.alive = True

    def reset(self, x, y):
        self.x, self.y = x, y
        self.angle = 0
        self.alive = True

    def update(self, action):
        if not self.alive:
            return
        self.angle += action
        rad = math.radians(self.angle)
        self.x += CAR_SPEED * math.sin(rad)
        self.y -= CAR_SPEED * math.cos(rad)

    def get_sensors(self, walls):
        """Возвращает список расстояний до ближайших стен"""
        readings = []
        start = (self.x, self.y)
        for d_angle in range(-90, 91, 180 // (NUM_SENSORS - 1)):
            ray_angle = math.radians(self.angle + d_angle)
            dx, dy = math.sin(ray_angle), -math.cos(ray_angle)
            for dist in range(0, SENSOR_RANGE, 5):
                end = (self.x + dx*dist, self.y + dy*dist)
                if any(pygame.Rect(wall).collidepoint(end) for wall in walls):
                    readings.append(dist)
                    break
            else:
                readings.append(SENSOR_RANGE)
        return readings

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 0, 0), (self.x-10, self.y-20, 20, 40))
