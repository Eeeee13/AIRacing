import pygame
import pymunk
import math
import numpy as np

class Car:
    def __init__(self, x, y, space):
        # Pygame часть
        self.original_image = pygame.image.load('blue_car.png')
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(x, y))
        self.angle = 0
        self.start_position = (1500, 850)
        
        # Pymunk часть
        mass = 50
        size = (self.rect.width * 0.8, self.rect.height * 0.6)
        moment = pymunk.moment_for_box(mass, size)
        
        self.body = pymunk.Body(mass, moment)
        self.body.position = x, y
        self.body.angle = math.radians(self.angle)
        
        self.shape = pymunk.Poly.create_box(self.body, size)
        self.shape.elasticity = 0.2
        self.shape.friction = 0.9

        self.original_mask = pygame.mask.from_surface(self.original_image)
        self.mask = self.original_mask
        self.angle = 0
        
        # Настройки управления
        self.steering_angle = 0  # Текущий угол поворота руля
        self.max_steering = 60   # Максимальный угол поворота (градусы)
        self.steering_speed = 10  # Скорость поворота руля
        self.wheel_base = 50     # База колес (расстояние между осями)


        self.ray_count = 6  # Количество лучей
        self.ray_angles = [-90, -45, 0, 45, 90, 180]  # Углы лучей относительно направления машины
        self.ray_distances = [0] * self.ray_count  # Расстояния до препятствий


        self.last_checkpoint = 0  # Последний пройденный чекпоинт
        self.checkpoint_positions = [  # Определи чекпоинты по трассе
            (1500, 500),   
            (1430, 210),  
            (1000, 200),
            (300, 200),
            (600, 420),
            (600, 630),
            (800, 860)
        ]
        self.lap_start_time = 0
        self.best_lap_time = float('inf')

        
        space.add(self.body, self.shape)

    def rotate(self, angle):
        self.angle = angle
        self.image = pygame.transform.rotate(self.original_image, -self.angle)
        
        self.mask = pygame.mask.from_surface(self.image)
        
        old_center = self.rect.center
        self.rect = self.image.get_rect()
        self.rect.center = old_center

    def reset_to_start(self):
        """Перемещает машину в стартовую позицию и сбрасывает состояние"""
        # Сбрасываем физическое тело
        self.body.position = self.start_position
        self.body.velocity = (0, 0)
        self.body.angular_velocity = 0
        self.body.angle = 0
        
        # Сбрасываем графику
        self.angle = 0
        self.image = self.original_image
        self.rect = self.image.get_rect(center=self.start_position)
        
        # Сбрасываем управление
        self.steering_angle = 0
        
        # Обновляем маску
        self.mask = pygame.mask.from_surface(self.image)
        self.last_checkpoint = 0

    
    def get_speed(self):
        """Получить текущую скорость"""
        velocity = self.body.velocity
        return math.sqrt(velocity.x**2 + velocity.y**2)
    
    def draw(self, screen):
        screen.blit(self.image, self.rect)
    
    def get_forward_speed(self):
        """Получить скорость вперед/назад"""
        # Проекция скорости на направление движения
        forward_vec = self.get_forward_vec()
        return np.dot([self.body.velocity.x, self.body.velocity.y], forward_vec)
    
    def get_forward_vec(self):
        """Вектор направления вперед"""
        angle = self.body.angle
        return [math.sin(angle), -math.cos(angle)]
    
    def update_steering(self, keys):
        """Обновление поворота руля"""
        # Накопление угла поворота руля
        if keys[pygame.K_RIGHT]:
            self.steering_angle += self.steering_speed
        elif keys[pygame.K_LEFT]:
            self.steering_angle -= self.steering_speed
        else:
            # Плавный возврат руля в центр
            if abs(self.steering_angle) > 0.5:
                self.steering_angle *= 0.9
            else:
                self.steering_angle = 0
        
        # Ограничение угла поворота
        self.steering_angle = max(-self.max_steering, min(self.max_steering, self.steering_angle))
    
    def apply_steering_force(self):
        speed = self.get_forward_speed()
        
        # Поворачиваем только если машина движется
        if abs(speed) > 1.0:
            # Рассчитываем силу поворота пропорционально скорости
            steering_power = 0.01 # Мощность поворота
            turn_force = speed * steering_power * self.steering_angle
            
            nose_offset = 30  # Смещение к передней части
            
           
            self.body.apply_force_at_local_point(
                (turn_force, 0),  # Сила в боковом направлении
                (0, nose_offset)  # Точка приложения - ближе к носу
            )
            # Дополнительно можно применить крутящий момент для более резкого поворота
            extra_torque = turn_force * 200  # Дополнительный момент
            self.body.torque += extra_torque
            
            # Боковое трение для реалистичности
            self.apply_lateral_friction()
        

    def apply_lateral_friction(self):
        """Применение бокового трения для реалистичного поворота"""
        # Получаем вектор направления движения
        forward_vec = self.get_forward_vec()
        # Получаем боковой вектор (перпендикулярно направлению)
        lateral_vec = [-forward_vec[1], forward_vec[0]]
        # Проекция скорости на боковое направление
        lateral_velocity = np.dot([self.body.velocity.x, self.body.velocity.y], lateral_vec)
        # Применяем боковое трение
        lateral_friction = -lateral_velocity * 0.4
        self.body.velocity = (
            self.body.velocity.x + lateral_vec[0] * lateral_friction,
            self.body.velocity.y + lateral_vec[1] * lateral_friction
        )
    
    def apply_engine_force(self, throttle):
        """Применение силы двигателя"""
        forward_vec = self.get_forward_vec()
        force = 9000 * throttle
        if self.get_speed() < 0:
            # self.body.apply_force_at_local_point((0, -force), (0, -30))
            pass
        else:
            self.body.apply_force_at_local_point((0, -force), (0, -30))

    def apply_friction(self):
        """Применение трения и сопротивления"""
        # Линейное трение (замедление)
        friction_force = 0.99 # Коэффициент трения (0.9-0.99)
        self.body.velocity = (
            self.body.velocity.x * friction_force,
            self.body.velocity.y * friction_force
        )
        
        # Угловое трение (замедление вращения)
        angular_friction = 0.97 # Коэффициент трения вращения
        self.body.angular_velocity *= angular_friction
        
        # Боковое трение (дрифт) - предотвращает боковое скольжение
        # self.apply_lateral_friction()
        
        # Полная остановка при очень малых скоростях
        if abs(self.body.velocity.x) < 0.1 and abs(self.body.velocity.y) < 0.1:
            self.body.velocity = (0, 0)
        if abs(self.body.angular_velocity) < 0.01:
            self.body.angular_velocity = 0

    def apply_lateral_friction(self):
        """Боковое трение для реалистичного поворота"""
        if abs(self.get_forward_speed()) > 2.0:  # Только при движении
            forward_vec = self.get_forward_vec()
            lateral_vec = [-forward_vec[1], forward_vec[0]]  # Перпендикулярно направлению
            
            # Проекция скорости на боковое направление
            lateral_speed = (
                self.body.velocity.x * lateral_vec[0] + 
                self.body.velocity.y * lateral_vec[1]
            )
            
            # Сила бокового трения (противоположная боковому скольжению)
            lateral_friction = -lateral_speed * 0.1  # Коэффициент бокового трения
            
            self.body.velocity = (
                self.body.velocity.x + lateral_vec[0] * lateral_friction,
                self.body.velocity.y + lateral_vec[1] * lateral_friction
            )
    
    def update(self, keys):

        self.update_steering(keys)
        
        # Применяем газ/тормоз
        throttle = 0
        if keys[pygame.K_UP]:
            throttle = 1.0
        elif keys[pygame.K_DOWN]:
            throttle = -2  # Задний ход/тормоз
        
        if throttle != 0:
            self.apply_engine_force(throttle)
        else:
            self.body.torque = 0
        
        # Применяем поворот только если машина движется
        self.apply_steering_force()

        self.apply_friction() # трение
        
        # Обновляем графику
        self.update_graphics()
    
    def update_graphics(self):
        """Синхронизация графики с физикой"""
        self.rect.center = (int(self.body.position.x), int(self.body.position.y))
        self.angle = math.degrees(self.body.angle)
        self.image = pygame.transform.rotate(self.original_image, -self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)

        # ОБНОВЛЯЕМ МАСКУ
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect(center=self.rect.center)
    
    def draw_steering_info(self, surface, font):
        """Отрисовка информации о повороте (для дебага)"""
        speed_text = font.render(f"Speed: {self.get_forward_speed():.1f}", True, (255, 255, 255))
        steering_text = font.render(f"Steering: {self.steering_angle:.1f}°", True, (255, 255, 255))
        surface.blit(speed_text, (10, 10))
        surface.blit(steering_text, (10, 40))



    def cast_rays(self, track):
        """Выпускает лучи и измеряет расстояние до границ трассы"""
        center = self.rect.center
        angle_rad = math.radians(self.angle)
        
        for i, ray_angle in enumerate(self.ray_angles):
            # Рассчитываем направление луча
            ray_dir = [
                math.sin(angle_rad + math.radians(ray_angle)),
                -math.cos(angle_rad + math.radians(ray_angle))
            ]
            
            # Выпускаем луч шаг за шагом
            max_distance = 300  # Максимальная длина луча
            distance = 0
            hit = False
            
            for step in range(1, max_distance):
                # Текущая позиция на луче
                pos = [
                    center[0] + ray_dir[0] * step,
                    center[1] + ray_dir[1] * step
                ]
                
                # Проверяем, находится ли позиция в пределах трассы
                if (0 <= pos[0] < track.rect.width and 
                    0 <= pos[1] < track.rect.height):
                    
                    # Проверяем столкновение с границей
                    if track.mask.get_at((int(pos[0]), int(pos[1]))):
                        distance = step
                        hit = True
                        break
                else:
                    break
            
            self.ray_distances[i] = distance if hit else max_distance


    def draw_rays(self, surface, track):
        """Отрисовывает лучи для визуализации"""
        center = self.rect.center
        angle_rad = math.radians(self.angle)
        
        for i, ray_angle in enumerate(self.ray_angles):
            if self.ray_distances[i] > 0:
                # Рассчитываем направление луча
                ray_dir = [
                    math.sin(angle_rad + math.radians(ray_angle)),
                    -math.cos(angle_rad + math.radians(ray_angle))
                ]
                
                # Конечная точка луча
                end_pos = [
                    center[0] + ray_dir[0] * self.ray_distances[i],
                    center[1] + ray_dir[1] * self.ray_distances[i]
                ]
                
                # Рисуем луч
                pygame.draw.line(surface, (255, 0, 0), center, end_pos, 1)


    def get_reward(self, crashed=False):
        """Вычисляет награду для агента"""
        if crashed:
            return -200.0  # Штраф за столкновение
        
        # Награда за скорость 
        velocity = math.sqrt(self.body.velocity.x**2 + self.body.velocity.y**2)
        speed_reward = velocity * 0.1  # Коэффициент подбери экспериментально
        
        # Награда за прохождение чекпоинтов
        checkpoint_reward = 0
        current_pos = self.body.position
        
        # Проверяем прохождение следующего чекпоинта
        next_checkpoint = self.checkpoint_positions[self.last_checkpoint % len(self.checkpoint_positions)]
        distance_to_checkpoint = math.sqrt((current_pos.x - next_checkpoint[0])**2 + 
                                        (current_pos.y - next_checkpoint[1])**2)
        
        if distance_to_checkpoint < 100:  # Если близко к чекпоинту
            self.last_checkpoint += 1
            checkpoint_reward = 100.0  # Большая награда за чекпоинт
            
            # Если прошли полный круг
            if self.last_checkpoint % len(self.checkpoint_positions) == 0:
                lap_time = pygame.time.get_ticks() - self.lap_start_time
                if lap_time < self.best_lap_time:
                    self.best_lap_time = lap_time
                    checkpoint_reward += 500.0  # Бонус за лучшее время
                self.lap_start_time = pygame.time.get_ticks()
        
        # Штраф за слишком медленное движение
        if velocity < 1.0:
            speed_reward -= 0.2
        
        return speed_reward + checkpoint_reward

