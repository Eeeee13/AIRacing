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
        self.start_position = (1530, 430)
        
        # Pymunk часть
        mass = 40
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
        self.max_steering = 80   # Максимальный угол поворота (градусы)
        self.steering_speed = 20  # Скорость поворота руля
        self.wheel_base = 50     # База колес (расстояние между осями)
        self.max_ray_distance = 300  # Максимальная длина луча


        self.ray_count = 5  # Количество лучей
        self.ray_angles = [-90, -45, 0, 45, 90]  # Углы лучей относительно направления машины
        self.ray_distances = [0] * self.ray_count  # Расстояния до препятствий


        self.last_checkpoint = 0  # Последний пройденный чекпоинт
        self.checkpoint_positions = [  # Определи чекпоинты по трассе
            # (1500, 500),   
            (1430, 210),  
            (1300, 210),
            (1000, 200),
            (750, 90),
            (530, 171),
            (300, 200),
            (600, 420),
            (600, 630),
            (800, 860)
        ]
        self.lap_start_time = 0
        self.best_lap_time = float('inf')

        
        space.add(self.body, self.shape)

###_____________Update_________###
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
    
    def reset(self):
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

    def draw(self, screen):
        screen.blit(self.image, self.rect)

###______________Reward___________####
    def compute_guided_reward(self):
        reward = 0
        if self.last_checkpoint + 1 == len(self.checkpoint_positions):
            next_checkpoint = self.checkpoint_positions[0]
        else:
            next_checkpoint = self.checkpoint_positions[self.last_checkpoint + 1]
        
        # # 1. Награда за направление к следующему чекпоинту
        # next_checkpoint = self.checkpoint_positions[self.last_checkpoint + 1]
        # direction_to_checkpoint = self.get_direction_to(next_checkpoint)
        # angle_diff = abs(self.angle - direction_to_checkpoint)
        # print("angle_diff: ",angle_diff)
        
        # # Награда за правильное направление (критически важно!)
        # reward += (1 - angle_diff/180) * 2  # Максимум +2 за идеальное направление
        # print("reward 1:", reward)
        
        # # 2. Награда за плавность траектории
        # if abs(self.steering_angle) < 10 and self.get_speed() > 1:
        #     reward += 0.5  # Награда за прямолинейное движение на скорости
        
        # 3. Прогрессивная награда за приближение к чекпоинту
        distance_to_checkpoint = self.get_distance_to(next_checkpoint)
        reward += (1 - distance_to_checkpoint/1000) * 0.2  # Увеличивается по мере приближения
        # print("reward 2:", reward)
        
        return reward
    
    def get_reward(self, crashed=False):
        """Вычисляет награду для агента"""
        speed = self.get_speed()
        forward_speed = self.get_forward_speed()
        reward = 0
        if crashed:
            reward  -= 10.0  # Штраф за столкновение

        # 1. ШТРАФ за движение назад
        if forward_speed < -0.5:  # Движение назад с заметной скоростью
            reward += forward_speed * 2  # Умножаем на 2 для усиления штрафа
            # Дополнительный фиксированный штраф
            reward -= 1.0
        
        current_pos = self.body.position
        
        # Проверяем прохождение следующего чекпоинта
        next_checkpoint = self.checkpoint_positions[self.last_checkpoint % len(self.checkpoint_positions)]
        distance_to_checkpoint = math.sqrt((current_pos.x - next_checkpoint[0])**2 + 
                                        (current_pos.y - next_checkpoint[1])**2)
        
        if distance_to_checkpoint < 100:  # Если близко к чекпоинту
            self.last_checkpoint += 1
            reward += 20.0  # Большая награда за чекпоинт
            
            # Если прошли полный круг
            if self.last_checkpoint % len(self.checkpoint_positions) == 0:
                lap_time = pygame.time.get_ticks() - self.lap_start_time
                if lap_time < self.best_lap_time:
                    self.best_lap_time = lap_time
                    reward += 500.0  # Бонус за лучшее время
                self.lap_start_time = pygame.time.get_ticks()
        
        # 3. Награда за СКОРОСТЬ - только вперед
        if forward_speed > 0:
            pass
            # reward += forward_speed * 0.01
        else:
            # Штраф за нулевую или отрицательную скорость
            # print("oh, no no no")
            reward -= 0.3
        
        return reward

###_____________observations_______#####
    def get_observations(self, track):
        observations = []
        
        # Критически важные наблюдения для поворотов:
        
        # 1. Относительное положение следующего чекпоинта
        if self.last_checkpoint + 1 == len(self.checkpoint_positions):
            next_cp = self.checkpoint_positions[0]
        else:
            next_cp = self.checkpoint_positions[self.last_checkpoint + 1]
        relative_x = next_cp[0] - self.rect.x
        relative_y = next_cp[1] - self.rect.y
        observations.extend([relative_x, relative_y])
        
        # 2. Угол до чекпоинта относительно направления машины
        angle_to_cp = self.get_angle_to(next_cp)
        angle_diff = angle_to_cp - self.angle
        observations.append(angle_diff)
        
        # # 3. Дистанции лучей (особенно боковых)
        # ray_distances = self.cast_rays(track)  # [левый, прямой, правый]
        # observations.extend(ray_distances)
        
        # 4. Текущая скорость и угол поворота
        observations.extend([self.get_speed(), self.steering_angle])
        
        return np.array(observations)
    
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

            distance = 0
            hit = False
            
            for step in range(1, self.max_ray_distance):
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
            
            self.ray_distances[i] = distance if hit else self.max_ray_distance
      

###_______________Helpers______________#####
    
    def rotate(self, angle):
        self.angle = angle
        self.image = pygame.transform.rotate(self.original_image, -self.angle)
        
        self.mask = pygame.mask.from_surface(self.image)
        
        old_center = self.rect.center
        self.rect = self.image.get_rect()
        self.rect.center = old_center
    
    def get_angle_to(self, next_cp):
        """
        расчет угла до чекпоинта.
        """
        car_x, car_y = self.body.position.x, self.body.position.y
        car_angle = math.degrees(self.body.angle)
        
        # Позиция чекпоинта
        cp_x, cp_y = next_cp

        dx = cp_x - car_x
        dy = cp_y - car_y
        
        # Угол к чекпоинту в системе координат Pygame
        # atan2 возвращает угол в радианах от оси X, нам нужно преобразовать
        angle_to_cp = math.degrees(math.atan2(-dy, dx)) - 90
        if angle_to_cp < 0:
            angle_to_cp += 360
        
        # Разница углов
        angle_diff = angle_to_cp - car_angle
        
        # Нормализуем в диапазон [-180, 180]
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        
        return angle_diff
    
    def get_distance_to(self, next_cp):
        """Расстояние до чекпоинта"""
        car_pos = np.array([self.body.position.x, self.body.position.y])
        cp_pos = np.array(next_cp)
        return np.linalg.norm(cp_pos - car_pos)
   
    def get_direction_to(self, next_cp):
        """
        Возвращает нормализованный вектор направления к чекпоинту.
        Полезно для дополнительных расчетов.
        """
        car_pos = np.array([self.body.position.x, self.body.position.y])
        cp_pos = np.array(next_cp)
        direction = cp_pos - car_pos
        if np.linalg.norm(direction) > 0:
            return direction / np.linalg.norm(direction)
        return np.array([0, 0])

    def get_forward_vec(self):
        """Вектор направления вперед"""
        angle = self.body.angle
        return [math.sin(angle), -math.cos(angle)]
    
    def get_forward_speed(self):
        """Получить скорость вперед/назад"""
        # Проекция скорости на направление движения
        forward_vec = self.get_forward_vec()
        return np.dot([self.body.velocity.x, self.body.velocity.y], forward_vec)
        
    def get_speed(self):
        """Получить текущую скорость"""
        velocity = self.body.velocity
        return math.sqrt(velocity.x**2 + velocity.y**2)


    def normalize_state(self, state):
        """Нормализует состояния для стабильности обучения"""
        state = np.array(state, dtype=np.float32)
        
        # Проверяем корректную длину состояния
        expected_length = 10  # 5 наблюдений + 5 лучей
        if len(state) != expected_length:
            print(f"Предупреждение: ожидалось {expected_length} элементов состояния, получено {len(state)}")
            # Дополняем нулями если не хватает элементов
            if len(state) < expected_length:
                state = np.pad(state, (0, expected_length - len(state)), 'constant')
            else:
                state = state[:expected_length]
        
        # Нормализация каждого компонента состояния
        normalized_state = state.copy()
        
        # Компоненты наблюдений (первые 5 элементов)
        if len(state) >= 5:
            # Относительные координаты чекпоинта (элементы 0, 1)
            normalized_state[0] = np.clip(state[0] / 1000.0, -5.0, 5.0)  # relative_x
            normalized_state[1] = np.clip(state[1] / 1000.0, -5.0, 5.0)  # relative_y
            
            # Угол до чекпоинта (элемент 2)
            normalized_state[2] = np.clip(state[2] / 180.0, -2.0, 2.0)   # angle_diff
            
            # Скорость (элемент 3)
            normalized_state[3] = np.clip(state[3] / 100.0, -2.0, 2.0)   # speed
            
            # Угол поворота (элемент 4)
            normalized_state[4] = np.clip(state[4] / self.max_steering, -1.0, 1.0)  # steering_angle
        
        # Расстояния лучей (элементы 5-9)
        if len(state) >= 10:
            for i in range(5, 10):
                normalized_state[i] = np.clip(state[i] / self.max_ray_distance, 0.0, 1.0)
        
        return normalized_state.astype(np.float32)
####______________debug_info__________#######

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

    def draw_steering_info(self, surface, font):
        """Отрисовка информации о повороте (для дебага)"""
        speed_text = font.render(f"Speed: {self.get_forward_speed():.1f}", True, (255, 255, 255))
        steering_text = font.render(f"Steering: {self.steering_angle:.1f}°", True, (255, 255, 255))
        surface.blit(speed_text, (10, 10))
        surface.blit(steering_text, (10, 40))

#######____________Physics_________________##########

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
            steering_power = 0.015
            turn_force = speed * steering_power * self.steering_angle
            
            nose_offset = 30  # Смещение к передней части
            
           
            self.body.apply_force_at_local_point(
                (turn_force, 0),  
                (0, nose_offset)  
            )
            # Дополнительно можно применить крутящий момент для более резкого поворота
            extra_torque = turn_force * 200  # Дополнительный момент
            self.body.torque += extra_torque
            
            # Боковое трение для реалистичности
            self.apply_lateral_friction()
        
    def apply_lateral_friction(self):
        """Применение бокового трения для реалистичного поворота"""
        forward_vec = self.get_forward_vec()
        lateral_vec = [-forward_vec[1], forward_vec[0]]
        lateral_velocity = np.dot([self.body.velocity.x, self.body.velocity.y], lateral_vec)
        lateral_friction = -lateral_velocity * 0.4
        self.body.velocity = (
            self.body.velocity.x + lateral_vec[0] * lateral_friction,
            self.body.velocity.y + lateral_vec[1] * lateral_friction
        )
    
    def apply_engine_force(self, throttle):
        """Применение силы двигателя"""
        # forward_vec = self.get_forward_vec()
        force = 7000 * throttle
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
    


