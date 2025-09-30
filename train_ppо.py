from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import os
import numpy as np
from racing_env import RacingEnv, make_racing_env
import matplotlib.pyplot as plt


def train_racing_agent():
    """Тренировка агента для гонок"""
    
    # Создаем директории для сохранения
    os.makedirs("./racing_logs/", exist_ok=True)
    os.makedirs("./racing_models/", exist_ok=True)
    
    # Создаем окружение
    # Для тренировки отключаем рендеринг для ускорения
    env = make_vec_env(lambda: make_racing_env(render_mode=None), n_envs=1)
    
    # Создаем отдельное окружение для оценки с рендерингом
    eval_env = Monitor(make_racing_env(render_mode="human"))
    
    # Настройки PPO
    model = PPO(
        "MlpPolicy",  # Многослойный перцептрон
        env,
        learning_rate=3e-4,
        n_steps=2048,           # Количество шагов для сбора данных
        batch_size=64,          # Размер батча для обучения
        n_epochs=10,            # Количество эпох обучения на собранных данных
        gamma=0.99,             # Коэффициент дисконтирования
        gae_lambda=0.95,        # GAE параметр
        clip_range=0.2,         # PPO clipping параметр
        clip_range_vf=None,     # Clipping для value function
        ent_coef=0.01,          # Коэффициент энтропии для исследования
        vf_coef=0.5,            # Коэффициент value function loss
        max_grad_norm=0.5,      # Максимальная норма градиента
        verbose=1,              # Уровень логирования
        tensorboard_log="./racing_logs/",
        device="auto"           # Автоматический выбор устройства (CPU/GPU)
    )
    
    # Настройка коллбеков
    # Остановка при достижении средней награды
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=1000, verbose=1)
    
    # Периодическая оценка модели
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=10000,        # Оценка каждые 10000 шагов
        best_model_save_path="./racing_models/",
        log_path="./racing_logs/",
        verbose=1
    )
    
    # Обучение модели
    print("Начинаем обучение...")
    model.learn(
        total_timesteps=500000,  # Общее количество шагов обучения
        callback=eval_callback,
        tb_log_name="PPO_racing"
    )
    
    # Сохраняем финальную модель
    model.save("./racing_models/ppo_racing_final")
    print("Обучение завершено! Модель сохранена.")
    
    return model


def test_trained_model(model_path="./racing_models/ppo_racing_final"):
    """Тестирование обученной модели"""
    
    # Создаем окружение для тестирования
    env = make_racing_env(render_mode="human")
    
    # Загружаем модель
    model = PPO.load(model_path)
    
    # Тестируем модель
    obs = env.reset()
    total_reward = 0
    steps = 0
    
    print("Тестирование модели...")
    print("Нажмите Ctrl+C для остановки")
    
    try:
        while True:
            # Получаем действие от модели
            action, _states = model.predict(obs, deterministic=True)
            
            # Выполняем действие
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Рендерим окружение
            env.render()
            
            if done:
                print(f"Эпизод завершен!")
                print(f"Общая награда: {total_reward:.2f}")
                print(f"Шагов: {steps}")
                print(f"Достигнут чекпоинт: {info['checkpoint']}")
                print(f"Столкновение: {info['crashed']}")
                print("-" * 50)
                
                # Сброс для нового эпизода
                obs = env.reset()
                total_reward = 0
                steps = 0
                
    except KeyboardInterrupt:
        print("Тестирование остановлено пользователем")
    
    env.close()


def continue_training(model_path="./racing_models/best_model", additional_steps=100000):
    """Продолжение обучения существующей модели"""
    
    # Создаем окружения
    env = make_vec_env(lambda: make_racing_env(render_mode=None), n_envs=1)
    eval_env = Monitor(make_racing_env(render_mode="human"))
    
    # Загружаем существующую модель
    model = PPO.load(model_path, env=env)
    
    # Настройка коллбеков
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=5000,
        best_model_save_path="./racing_models/",
        log_path="./racing_logs/",
        verbose=1
    )
    
    print(f"Продолжаем обучение на {additional_steps} шагов...")
    
    # Продолжаем обучение
    model.learn(
        total_timesteps=additional_steps,
        callback=eval_callback,
        reset_num_timesteps=False,  # Не сбрасываем счетчик шагов
        tb_log_name="PPO_racing_continued"
    )
    
    # Сохраняем обновленную модель
    model.save("./racing_models/ppo_racing_continued")
    print("Дообучение завершено!")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Обучение PPO агента для гонок")
    parser.add_argument("--mode", choices=["train", "test", "continue"], 
                       default="continue", help="Режим: train, test или continue")
    parser.add_argument("--model_path", default="./racing_models/best_model", 
                       help="Путь к модели для тестирования или продолжения обучения")
    parser.add_argument("--steps", type=int, default=100000, 
                       help="Дополнительные шаги для продолжения обучения")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        model = train_racing_agent()
    elif args.mode == "test":
        test_trained_model(args.model_path)
    elif args.mode == "continue":
        model = continue_training(args.model_path, args.steps)
    
    print("Готово!")