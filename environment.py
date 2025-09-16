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



import math
import random
import numpy as np
from scipy.interpolate import splprep, splev

def generate_track(points, road_width=ROAD_WIDTH, smoothness=0.5):
    """
    points: список (x,y) контрольных точек центра дороги
    road_width: ширина дороги
    smoothness: степень сглаживания (0-1)
    Возвращает: walls (список отрезков), road_polygon (для отрисовки)
    """
    if len(points) < 3:
        # Автоматически генерируем интересную трассу если точек мало
        points = generate_interesting_track()
    
    # Сглаживаем трассу с помощью сплайнов
    points = smooth_track(points, smoothness)
    
    half_w = road_width // 2
    left_side = []
    right_side = []

    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        x_prev, y_prev = points[(i - 1) % len(points)]

        # Вычисляем нормаль с учетом предыдущей и следующей точек
        if i == 0:
            # Для первой точки используем только следующую точку
            dx, dy = x2 - x1, y2 - y1
        else:
            # Усредняем нормали от предыдущего и следующего сегментов
            dx1, dy1 = x1 - x_prev, y1 - y_prev
            dx2, dy2 = x2 - x1, y2 - y1
            dx = (dx1 + dx2) / 2
            dy = (dy1 + dy2) / 2

        length = math.hypot(dx, dy)
        if length == 0:
            continue

        # Нормаль к направлению движения (перпендикуляр)
        nx, ny = -dy / length, dx / length

        # Смещаем влево и вправо
        left_side.append((x1 + nx * half_w, y1 + ny * half_w))
        right_side.append((x1 - nx * half_w, y1 - ny * half_w))

    # Формируем стены (отрезки)
    walls = []
    for i in range(len(left_side)):
        p1 = left_side[i]
        p2 = left_side[(i + 1) % len(left_side)]
        walls.append((p1, p2))
        
        p1 = right_side[i]
        p2 = right_side[(i + 1) % len(right_side)]
        walls.append((p1, p2))

    # Полигон дороги (соединяем левую и правую сторону)
    road_polygon = left_side + right_side[::-1]

    return walls, road_polygon

def generate_interesting_track(num_points=8, size_ratio=0.7):
    """
    Автоматически генерирует интересную трассу с поворотами
    """
    points = []
    
    # Центр экрана
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    track_radius = min(WIDTH, HEIGHT) * size_ratio // 2
    
    # Создаем точки на эллипсе с некоторой случайностью
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        
        # Добавляем случайность для создания интересной формы
        radius_variation = random.uniform(0.7, 1.3)
        angle_offset = random.uniform(-0.2, 0.2)
        
        effective_angle = angle + angle_offset
        radius = track_radius * radius_variation
        
        x = center_x + radius * math.cos(effective_angle)
        y = center_y + radius * math.sin(effective_angle)
        
        points.append((x, y))
    
    # Добавляем S-образные повороты
    points = add_s_shapes(points)
    
    return points

def smooth_track(points, smoothness=0.5):
    """
    Сглаживает трассу с помощью B-сплайнов
    """
    if len(points) < 4:
        return points
    
    # Преобразуем точки в numpy array
    points_array = np.array(points)
    x = points_array[:, 0]
    y = points_array[:, 1]
    
    # Замыкаем трассу
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    
    # Параметр сглаживания
    s = len(points) * (1 - smoothness) * 10
    
    try:
        # Создаем сплайн
        tck, u = splprep([x, y], s=s, per=True)
        
        # Генерируем сглаженные точки
        u_new = np.linspace(0, 1, len(points) * 3)
        x_new, y_new = splev(u_new, tck)
        
        # Преобразуем обратно в список точек
        smoothed_points = list(zip(x_new, y_new))
        
        # Убедимся, что первая и последняя точки совпадают
        if len(smoothed_points) > 0:
            smoothed_points[-1] = smoothed_points[0]
            
        return smoothed_points
        
    except:
        # Если сглаживание не удалось, возвращаем исходные точки
        return points

def add_s_shapes(points, intensity=0.3):
    """
    Добавляет S-образные повороты для большей сложности
    """
    if len(points) < 6:
        return points
    
    new_points = []
    
    for i in range(len(points)):
        x, y = points[i]
        
        if i % 3 == 0 and i > 0 and i < len(points) - 2:
            # Добавляем дополнительную точку для S-поворота
            next_x, next_y = points[i + 1]
            prev_x, prev_y = points[i - 1]
            
            # Вектор направления
            dx = next_x - prev_x
            dy = next_y - prev_y
            length = math.hypot(dx, dy)
            
            if length > 0:
                # Перпендикулярный вектор
                perp_x = -dy / length * intensity * 50
                perp_y = dx / length * intensity * 50
                
                # Добавляем смещенную точку
                new_points.append((x + perp_x, y + perp_y))
        
        new_points.append((x, y))
    
    return new_points

# Пример использования:
# def create_sample_track():
    # """Создает пример интересной трассы"""
    # center_x, center_y = WIDTH // 2, HEIGHT // 2
    
    # # Контрольные точки для трассы
    # control_points = [
    #     (center_x - 100, center_y - 150),
    #     (center_x + 200, center_y - 100),
    #     (center_x + 250, center_y + 100),
    #     (center_x, center_y + 200),
    #     (center_x - 250, center_y + 100),
    #     (center_x - 200, center_y - 100),
    # ]
    
    # walls, road_polygon, smoothed_points = generate_track(control_points, ROAD_WIDTH, 0.7)
    # return walls, road_polygon, smoothed_points

class Environment:
    def __init__(self):
        self.center_x, self.center_y = WIDTH // 2, HEIGHT // 2
    
        # Контрольные точки для трассы
        self.track_points = [
            (self.center_x - 100, self.center_y - 150),
            (self.center_x + 200, self.center_y - 100),
            (self.center_x + 250, self.center_y + 100),
            (self.center_x, self.center_y + 200),
            (self.center_x - 250, self.center_y + 100),
            (self.center_x - 200, self.center_y - 100),
    ]
        

        self.walls, self.road_polygon = generate_track(self.track_points, road_width=120)
        self.car = Car()

    def reset(self):
        self.car.reset(WIDTH//2, HEIGHT-100)
        return self.car.get_state(self.walls)
    

    def step(self, action):
        self.car.update(action)

        # Проверка: внутри ли машина трассы
        if not point_in_polygon((self.car.x, self.car.y), self.road_polygon):
            self.car.alive = False
            reward = -100
            done = True
            return self.car.get_state(self.walls), reward, done

        # Если жива → получаем сенсоры и награду
        sensors = self.car.get_state(self.walls)
        reward = 1  # базовая награда за выживание
        done = False

        return sensors, reward, done


    def render(self, screen):
        screen.fill((50, 200, 50))  # трава
        pygame.draw.polygon(screen, (100, 100, 100), self.road_polygon)  # дорога
        for wall in self.walls:
            pygame.draw.line(screen, (255, 255, 255), wall[0], wall[1], 3)
        self.car.draw(screen)

