import pygame
import random
import math
import numpy as np

# --- параметры игры ---
WIDTH, HEIGHT = 800, 1000
ROAD_WIDTH = 200
FPS = 60

# --- параметры машины ---
CAR_SPEED = 5
ANGLE_STEP = 1  # шаг дискретизации
ACTIONS = [-10, 0, 10]  # влево, прямо, вправо

# --- Q-learning ---
Q = {}  # Q[угол][действие]
alpha = 0.2  # скорость обучения
gamma = 0.5  # коэффициент дисконтирования
epsilon = 0.1  # вероятность случайного действия

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

class Car:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT - 100
        self.angle = 0
        self.alive = True

    def reset(self):
        self.x = WIDTH // 2
        self.y = HEIGHT - 100
        self.angle = 0
        self.alive = True

    def update(self, action):
        if not self.alive:
            return
        
        self.angle += action
        rad = math.radians(self.angle)
        self.x += CAR_SPEED * math.sin(rad)
        self.y -= CAR_SPEED * math.cos(rad)

        # проверка выхода за дорогу
        if self.x < (WIDTH // 2 - ROAD_WIDTH // 2) or self.x > (WIDTH // 2 + ROAD_WIDTH // 2):
            self.alive = False

    def draw(self):
        pygame.draw.rect(screen, (255, 0, 0), (self.x - 10, self.y - 20, 20, 40))

def discretize(angle):
    return round(angle / ANGLE_STEP) * ANGLE_STEP

def get_action(state):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    else:
        if state not in Q:
            Q[state] = {a: 0 for a in ACTIONS}
        return max(Q[state], key=Q[state].get)

def update_Q(state, action, reward, next_state):
    if state not in Q:
        Q[state] = {a: 0 for a in ACTIONS}
    if next_state not in Q:
        Q[next_state] = {a: 0 for a in ACTIONS}
    best_next = max(Q[next_state].values())
    Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])

# --- игра ---
car = Car()
running = True
reward = 0

while running:
    screen.fill((50, 200, 50))  # трава
    pygame.draw.rect(screen, (100, 100, 100), (WIDTH//2 - ROAD_WIDTH//2, 0, ROAD_WIDTH, HEIGHT))  # дорога

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # print("game over")
            running = False

    state = discretize(car.angle)
    action = get_action(state)

    car.update(action)
    car.draw()

    # награда
    if car.alive:
        reward = 1
    else:
        reward = -100
        car.reset()

    next_state = discretize(car.angle)
    update_Q(state, action, reward, next_state)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
