import pygame
import sys
import numpy as np
import pymunk
import pymunk.pygame_util
import math
import matplotlib.pyplot as plt
from car_ import Car 
from enviroment import Track
from PPO import PPOAgent

pygame.init()
space = pymunk.Space()
space.gravity = (0, 0)
screen = pygame.display.set_mode((1670, 1000))
draw_options = pymunk.pygame_util.DrawOptions(screen)
clock = pygame.time.Clock()
pygame.display.set_caption("Гонки на Python - PPO Training")
background = pygame.image.load('road.png')
font = pygame.font.Font(None, 36)
track = Track('road.png')
track.create_mask_from_color(background)

car = Car(400, 300, space)

# Инициализация PPO агента
agent = PPOAgent(state_size=6, action_size=2, lr=1e-4)

# Параметры тренировки
TRAINING_MODE = True  # Переключатель: True для тренировки, False для игры человеком
RENDER_TRAINING = True  # Показывать ли визуализацию во время тренировки
UPDATE_FREQUENCY = 5000  # Частота обновления агента (каждые N шагов)
MAX_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 2000

# Статистика
episode_rewards = []
episode = 0
step_count = 0
episode_reward = 0
episode_steps = 0

def normalize_ray_distances(ray_distances, max_distance=300):
    """Нормализация расстояний лучей для нейросети"""
    return np.array(ray_distances) / max_distance

def apply_action_to_car(action, car):
    """Применяет действие агента к машине"""
    throttle, steering_angle = action
    
    # Применяем газ/тормоз
    if throttle > 0:
        car.apply_engine_force(throttle)
    elif throttle < 0:
        car.apply_engine_force(throttle)  # Торможение
    else:
        car.body.torque = 0
    
    # Применяем поворот руля
    car.steering_angle = max(-60, min(60, steering_angle))  # Ограничиваем угол
    car.apply_steering_force()

# Попытка загрузить существующую модель
try:
    agent.load('ppo_car_model.pth')
    print("Модель загружена!")
except:
    print("Модель не найдена, начинаем с нуля")

running = True
paused = False

while running and episode < MAX_EPISODES:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_t:
                TRAINING_MODE = not TRAINING_MODE
                print(f"Режим: {'Тренировка' if TRAINING_MODE else 'Человек'}")
            elif event.key == pygame.K_s:
                agent.save('ppo_car_model.pth')
                print("Модель сохранена!")
    
    if paused:
        continue
    
    # Получаем состояние (расстояния лучей)
    car.cast_rays(track)
    state = normalize_ray_distances(car.ray_distances)
    
    if TRAINING_MODE:
        # Получаем действие от агента
        action, log_prob, value = agent.get_action(state)
        apply_action_to_car(action, car)
        
        # ВАЖНО: обновляем графику машины после применения действий
        car.apply_friction()  # применяем трение
        car.update_graphics()  # обновляем графическое представление
    else:
        # Управление человеком
        keys = pygame.key.get_pressed()
        car.update(keys)

    # Обновляем физику
    space.step(1/60)
    
    # Проверяем столкновение и вычисляем награду
    crashed = track.check_collision(car.mask, (car.rect.x, car.rect.y))
    reward = car.get_reward(crashed=crashed)
    episode_reward += reward
    episode_steps += 1
    step_count += 1
    
    # Определяем конец эпизода
    done = crashed or episode_steps >= MAX_STEPS_PER_EPISODE
    
    if TRAINING_MODE:
        # Сохраняем переход в буфере агента
        agent.store_transition(state, action, reward, log_prob, value, done)
        
        # Обновляем агента периодически
        if step_count % UPDATE_FREQUENCY == 0:
            print(f"Обновление агента на шаге {step_count}")
            agent.update()
    
    # Если эпизод завершен
    if done:
        if TRAINING_MODE:
            episode_rewards.append(episode_reward)
            print(f"Эпизод {episode}: Награда = {episode_reward:.2f}, Шагов = {episode_steps}")
            
            # Построение графика каждые 50 эпизодов
            if episode % 50 == 0 and len(episode_rewards) > 1:
                plt.figure(figsize=(10, 6))
                plt.plot(episode_rewards)
                plt.title('Награда по эпизодам')
                plt.xlabel('Эпизод')
                plt.ylabel('Общая награда')
                plt.grid(True)
                plt.savefig(f'training_progress_episode_{episode}.png')
                plt.close()
        
        # Сброс для нового эпизода
        car.reset_to_start()
        episode_reward = 0
        episode_steps = 0
        episode += 1
    
    # Отрисовка (если включена во время тренировки или в режиме игры)
    if not TRAINING_MODE or RENDER_TRAINING:
        screen.blit(background, (0, 0))
        car.draw(screen)
        
        # Отображаем информацию
        if TRAINING_MODE:
            info_text = [
                f"Эпизод: {episode}",
                f"Шаги: {episode_steps}",
                f"Награда: {episode_reward:.2f}",
                f"Скорость: {math.sqrt(car.body.velocity.x**2 + car.body.velocity.y**2):.1f}",
                "ПРОБЕЛ - пауза, T - переключить режим, S - сохранить"
            ]
        else:
            info_text = [
                "Режим: Человек",
                "T - переключить на ИИ",
                "Стрелки - управление"
            ]
        
        for i, text in enumerate(info_text):
            text_surface = font.render(text, True, (255, 255, 255))
            screen.blit(text_surface, (10, 10 + i * 30))
        
        car.draw_steering_info(screen, font)
        car.draw_rays(screen, track)
        
        pygame.display.flip()
    
    # Ограничиваем FPS только если рендеринг включен
    if not TRAINING_MODE or RENDER_TRAINING:
        clock.tick(60)

# Финальное сохранение модели
if TRAINING_MODE:
    agent.save('ppo_car_model_final.pth')
    
    # Построение финального графика
    plt.figure(figsize=(12, 8))
    plt.plot(episode_rewards)
    plt.title('Прогресс тренировки PPO агента')
    plt.xlabel('Эпизод')
    plt.ylabel('Общая награда за эпизод')
    plt.grid(True)
    
    # Скользящее среднее для сглаживания
    if len(episode_rewards) > 10:
        window_size = min(50, len(episode_rewards) // 10)
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(episode_rewards)), moving_avg, 
                color='red', label=f'Скользящее среднее ({window_size})')
        plt.legend()
    
    plt.savefig('final_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

pygame.quit()
sys.exit()