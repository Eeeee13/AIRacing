# config.py
# базовые параметры (дублирую только важные, остальное как у тебя)
WIDTH, HEIGHT = 800, 600
ROAD_WIDTH = 200
FPS = 60

CAR_SPEED = 5
SENSOR_RANGE = 150
NUM_SENSORS = 5

# дискретизация угла оставляем, но сеть будет работать с непрерывными сенсорами
ACTIONS = [-10, 0, 10]  # повороты (в градусах)
ACTION_SPACE = len(ACTIONS)

# DQN-параметры
DQN_BATCH_SIZE = 64
DQN_GAMMA = 0.99
DQN_LR = 1e-3
DQN_EPS_START = 1.0
DQN_EPS_END = 0.05
DQN_EPS_DECAY = 20000  # шагов до уменьшения eps
DQN_TARGET_UPDATE = 1000  # шагов — обновлять target сетку
DQN_REPLAY_SIZE = 100000
DQN_MIN_REPLAY = 1000  # минимум в реплее перед обучением
DQN_DEVICE = "cpu"  # или "cuda" если доступно

# прочее
SEED = 0
