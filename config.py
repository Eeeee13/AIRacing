# config.py
ROAD_WIDTH = 100
FPS = 60

CAR_SPEED = 5
SENSOR_RANGE = 150
NUM_SENSORS = 5

# дискретизация угла оставляем, но сеть будет работать с непрерывными сенсорами
ACTIONS = [0, 1, 2, 3, 4]  
# 0=ничего, 1=газ, 2=тормоз, 3=влево+газ, 4=вправо+газ

ACTION_SPACE = len(ACTIONS)

WIDTH, HEIGHT = 800, 600
SENSOR_COUNT = 8
MAX_SPEED = 6.0
ACCELERATION = 0.2
FRICTION = 0.05
TURN_SPEED = 4.0

STATE_DIM = SENSOR_COUNT + 3   # сенсоры + скорость + ускорение + угол руля
ACTION_DIM = 2                 # steer, throttle


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
