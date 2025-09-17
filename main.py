import pygame
import sys

pygame.init()
screen = pygame.display.set_mode((1670, 1000))
pygame.display.set_caption("Гонки на Python")
background = pygame.image.load('road.png')


class Car:
    def __init__(self, x, y):
        self.image = pygame.image.load('red_car.png')
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)
        self.speed = 1

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    def move(self, dx, dy):
        self.rect.x += dx
        self.rect.y += dy    



running = True
car = Car(400, 500)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        car.move(-car.speed, 0)
    if keys[pygame.K_RIGHT]:
        car.move(car.speed, 0)
    if keys[pygame.K_UP]:
        car.move(0, -car.speed)
    if keys[pygame.K_DOWN]:
        car.move(0, car.speed)

    screen.blit(background, (0, 0))
    car.draw(screen)
    pygame.display.flip()

pygame.quit()
sys.exit()