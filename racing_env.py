import gymnasium as gym
import numpy as np
import pygame
import pymunk
import math
from gymnasium import spaces
from car_ import Car
from enviroment import Track


class RacingEnv(gym.Env):
    """Кастомное окружение для обучения вождению с помощью PPO"""
    
    def __init__(self, render_mode="human"):
        super(RacingEnv, self).__init__()
        
        self.render_mode = render_mode
        
        # Инициализация pygame и pymunk
        pygame.init()
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        
        # Параметры экрана
        self.screen_width = 1670
        self.screen_height = 1000

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        
        if self.render_mode == "human":
            
            pygame.display.set_caption("Racing RL Training")
            self.clock = pygame.time.Clock()
        else:
            # self.screen = None
            pass
            
        # Загружаем фон и создаем трассу
        self.background = pygame.image.load('road.png')
        self.track = Track('road.png')
        self.track.create_mask_from_color(self.background)
        
        # Создаем машину
        self.car = Car(1530, 430, self.space)  # Стартовая позиция
        
        # Определяем пространство действий: [throttle, steering]
        # throttle: -1.0 (назад) до 1.0 (вперед)
        # steering: -1.0 (лево) до 1.0 (право)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        # Определяем пространство состояний
        # [относительные координаты чекпоинта (2), угол до чекпоинта (1), 
        #  скорость (1), угол поворота (1), расстояния лучей (5)]
        obs_size = 10  # 2 + 1 + 1 + 1 + 5
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Параметры симуляции
        self.max_steps = 2048
        self.current_step = 0
        
        # Для отслеживания прогресса
        self.last_distance_to_checkpoint = float('inf')
        self.steps_without_progress = 0
        self.max_steps_without_progress = 300
        
        self.font = None
        if self.render_mode == "human":
            self.font = pygame.font.Font(None, 36)

    def reset(self, seed=None, options=None):
        """Сброс окружения к начальному состоянию"""
        super().reset(seed=seed)
        # Сбрасываем машину
        self.car.reset()
        
        # Сбрасываем счетчики
        self.current_step = 0
        self.last_distance_to_checkpoint = float('inf')
        self.steps_without_progress = 0
        
        # Возвращаем начальное состояние
        return self._get_observation(), {}
    
    def step(self, action):
        """Выполнение одного шага в окружении"""
        self.current_step += 1
        
        # Применяем действие к машине
        self._apply_action(action)
        
        # Обновляем физику
        self.car.apply_friction()
        self.car.update_graphics()
        self.space.step(1/60)
        
        # Получаем новое состояние
        obs = self._get_observation()
        
        # Вычисляем награду
        reward = self._calculate_reward()
        
        # Проверяем условия завершения эпизода
        done = self._check_done()
        
        # Дополнительная информация
        info = {
            'checkpoint': self.car.last_checkpoint,
            'speed': self.car.get_speed(),
            'crashed': self.track.check_collision(self.car.mask, (self.car.rect.x, self.car.rect.y))
        }
        truncared = False

        assert not np.isnan(obs).any(), "NaN in observation"
        assert np.isfinite(obs).all(), "Inf in observation"
        assert np.isfinite(reward), "Invalid reward"

        
        return obs, reward, done, truncared, info
    
    def render(self, mode="human"):
        """Отрисовка окружения"""
        if self.render_mode != "human" or self.screen is None:
            return
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        
        # Очищаем экран и рисуем фон
        self.screen.blit(self.background, (0, 0))
        
        # Рисуем машину
        self.car.draw(self.screen)
        
        # Рисуем лучи для отладки
        self.car.draw_rays(self.screen, self.track)

        self.car.draw_checkpoints(self.screen)
        
        # Отображаем информацию
        if self.font:
            info_texts = [
                f"Шаг: {self.current_step}",
                f"Чекпоинт: {self.car.last_checkpoint}",
                f"Скорость: {self.car.get_speed():.1f}",
                f"Позиция: ({int(self.car.body.position.x)}, {int(self.car.body.position.y)})"
            ]
            
            for i, text in enumerate(info_texts):
                text_surface = self.font.render(text, True, (255, 255, 255))
                self.screen.blit(text_surface, (10, 10 + i * 30))
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def close(self):
        """Закрытие окружения"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
    
    def _get_observation(self):
        """Получение текущего состояния окружения"""
        # Обновляем лучи
        self.car.cast_rays(self.track)
        
        # Получаем базовые наблюдения от машины
        obs = self.car.get_observations(self.track)
        
        # Добавляем расстояния лучей
        ray_distances = np.array(self.car.ray_distances, dtype=np.float32)
        
        # Объединяем все наблюдения
        full_obs = np.concatenate([obs, ray_distances])
        
        # Нормализуем 
        normalized_obs = self.car.normalize_state(full_obs)
        return normalized_obs.astype(np.float32)
    
    def _apply_action(self, action):
        """Применение действия агента к машине"""
        throttle, steering = action
        
        # Масштабируем действия
        throttle = float(throttle)  # уже в диапазоне [-1, 1]
        steering_angle = float(steering) * self.car.max_steering  # масштабируем до максимального угла
        
        # Применяем газ/тормоз
        if abs(throttle) > 0.1:  # Мертвая зона
            self.car.apply_engine_force(throttle)
        else:
            self.car.body.torque = 0
        
        # Применяем поворот
        self.car.steering_angle = np.clip(steering_angle, -self.car.max_steering, self.car.max_steering)
        self.car.apply_steering_force()
    
    def _calculate_reward(self):
        """Вычисление награды за текущий шаг"""
        reward = 0.0
        
        # Проверяем столкновение
        crashed = self.track.check_collision(self.car.mask, (self.car.rect.x, self.car.rect.y))
        
        # Базовая награда от машины
        reward += self.car.get_reward(crashed=crashed)
        
        # Дополнительная направленная награда
        reward += self.car.compute_guided_reward()
        
        # Награда за прогресс к чекпоинту
        next_checkpoint = self.car.checkpoint_positions[self.car.last_checkpoint % len(self.car.checkpoint_positions)]
        current_distance = self.car.get_distance_to(next_checkpoint)
        
        # # Если приближаемся к чекпоинту
        # if current_distance < self.last_distance_to_checkpoint:
        #     reward += (self.last_distance_to_checkpoint - current_distance) * 0.01
        #     self.steps_without_progress = 0
        # else:
        #     self.steps_without_progress += 1
        
        # self.last_distance_to_checkpoint = current_distance
        
        
        return float(reward)
    
    def _check_done(self):
        """Проверка условий завершения эпизода"""
        # Столкновение
        if self.track.check_collision(self.car.mask, (self.car.rect.x, self.car.rect.y)):
            return True
        
        # Превышено максимальное количество шагов
        if self.current_step >= self.max_steps:
            return True
        
        # Слишком долго без прогресса
        if self.steps_without_progress >= self.max_steps_without_progress:
            return True
        
        # Машина остановилась надолго
        if self.car.get_speed() < 0.5 and self.current_step > 100:
            return True
        
        return False


# Функция для создания окружения (нужна для некоторых версий stable-baselines3)
def make_racing_env(render_mode="human"):
    return RacingEnv(render_mode=render_mode)