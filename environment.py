import pygame
from car import Car
from config import WIDTH, HEIGHT, ROAD_WIDTH

class Environment:
    def __init__(self):
        self.car = Car(WIDTH//2, HEIGHT-100)
        self.walls = [
            pygame.Rect(WIDTH//2 - ROAD_WIDTH//2, 0, 10, HEIGHT),  # левая
            pygame.Rect(WIDTH//2 + ROAD_WIDTH//2, 0, 10, HEIGHT),  # правая
        ]

    def reset(self):
        self.car.reset(WIDTH//2, HEIGHT-100)
        return self.car.get_sensors(self.walls)

    def step(self, action):
        self.car.update(action)
        sensors = self.car.get_sensors(self.walls)
        reward = 1
        done = False

        if not (self.walls[0].right < self.car.x < self.walls[1].left):
            reward = -100
            done = True

        return sensors, reward, done

    def render(self, screen):
        screen.fill((50, 200, 50))
        pygame.draw.rect(screen, (100,100,100), (WIDTH//2 - ROAD_WIDTH//2, 0, ROAD_WIDTH, HEIGHT))
        for wall in self.walls:
            pygame.draw.rect(screen, (0,0,0), wall)
        self.car.draw(screen)
