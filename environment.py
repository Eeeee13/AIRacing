import pygame
from car import Car
from config import WIDTH, HEIGHT, ROAD_WIDTH
import math

def point_in_polygon(point, polygon):
    """Проверяет, находится ли точка внутри полигона (ray casting algorithm)."""
    x, y = point
    inside = False
    n = len(polygon)
    p1x, p1y = polygon[0]

    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside



def generate_track(points, road_width=100):
        """
        points: список (x,y) контрольных точек центра дороги
        road_width: ширина дороги
        Возвращает: walls (список отрезков), road_polygon (для отрисовки)
        """
        half_w = road_width // 2
        left_side = []
        right_side = []

        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]  # замыкаем в цикл

            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length == 0:
                continue

            # нормаль к отрезку (влево/вправо)
            nx, ny = -dy / length, dx / length

            # смещаем влево и вправо
            left_side.append((x1 + nx * half_w, y1 + ny * half_w))
            right_side.append((x1 - nx * half_w, y1 - ny * half_w))

        # формируем стены (отрезки)
        walls = []
        for i in range(len(left_side)):
            walls.append((left_side[i], left_side[(i + 1) % len(left_side)]))
            walls.append((right_side[i], right_side[(i + 1) % len(right_side)]))

        # полигон дороги (соединяем левую и правую сторону)
        road_polygon = left_side + right_side[::-1]

        return walls, road_polygon

class Environment:
    def __init__(self):
        self.track_points = [
            (200, 200),
            (600, 200),
            (700, 400),
            (600, 500),
            (200, 500),
            (100, 400)
        ]

        self.walls, self.road_polygon = generate_track(self.track_points, road_width=120)
        self.car = Car(0, 0)

    def reset(self):
        self.car.reset(WIDTH//2, HEIGHT-100)
        return self.car.get_sensors(self.walls)
    

    def step(self, action):
        self.car.update(action)

        # Проверка: внутри ли машина трассы
        if not point_in_polygon((self.car.x, self.car.y), self.road_polygon):
            self.car.alive = False
            reward = -100
            done = True
            return self.car.get_sensors(self.walls), reward, done

        # Если жива → получаем сенсоры и награду
        sensors = self.car.get_sensors(self.walls)
        reward = 1  # базовая награда за выживание
        done = False

        return sensors, reward, done


    def render(self, screen):
        screen.fill((50, 200, 50))  # трава
        pygame.draw.polygon(screen, (100, 100, 100), self.road_polygon)  # дорога
        for wall in self.walls:
            pygame.draw.line(screen, (255, 255, 255), wall[0], wall[1], 3)
        self.car.draw(screen)

