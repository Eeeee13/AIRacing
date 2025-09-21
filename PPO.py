import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

class PPONetwork(nn.Module):
    def __init__(self, state_size=6, action_size=2, hidden_size=128):
        super(PPONetwork, self).__init__()
        
        # Общие слои
        self.shared_fc1 = nn.Linear(state_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)
        
        # политика
        self.actor_mean = nn.Linear(hidden_size, action_size)
        self.actor_std = nn.Parameter(torch.ones(action_size) * 0.5)
        
        # Critic функция ценности
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, state):
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))
        
        # Actor выход
        action_mean = torch.tanh(self.actor_mean(x))  # [-1, 1] диапазон
        action_std = F.softplus(self.actor_std) + 1e-5  # Обеспечиваем положительное значение
        
        # Critic выход
        value = self.critic(x)
        
        return action_mean, action_std, value

class PPOAgent:
    def __init__(self, state_size=6, action_size=2, lr=3e-4, gamma=0.99, eps_clip=0.2, 
                 k_epochs=4, entropy_coef=0.01):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PPONetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        
        # Буферы для хранения опыта
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
    def get_action(self, state):
        """Получить действие от агента"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_mean, action_std, value = self.network(state)
            
        # Создаем нормальное распределение
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Преобразуем действия в нужный диапазон
        # action[0] - газ/тормоз: [-1, 1] -> [-2, 1] (тормоз сильнее)
        # action[1] - поворот: [-1, 1] -> [-60, 60] градусов
        processed_action = action.cpu().numpy()[0]
        
        # Газ/тормоз: -1 до 1, где отрицательные значения - тормоз
        throttle = processed_action[0]
        if throttle < 0:
            throttle *= 2  # Усиливаем торможение
        
        # Поворот руля
        steering = processed_action[1] * 60  # [-60, 60] градусов
        
        return [throttle, steering], log_prob.cpu().numpy(), value.cpu().numpy()[0, 0]
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """Сохранить переход в буфер"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_returns(self):
        """Вычислить дисконтированные возвраты"""
        returns = []
        discounted_reward = 0
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
            
        return returns
    
    def update(self):
        """Обновить политику используя PPO"""
        if len(self.states) == 0:
            return
        
        # Вычисляем возвраты
        returns = self.compute_returns()
        
        # Конвертируем в тензоры
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        old_values = torch.FloatTensor(self.values).to(self.device)
        
        # Нормализация возвратов
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Вычисляем преимущества
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO обновления
        for _ in range(self.k_epochs):
            # Получаем новые предсказания
            action_mean, action_std, values = self.network(states)
            
            # Вычисляем новые логарифмы вероятностей
            dist = Normal(action_mean, action_std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
            # Вычисляем отношение вероятностей
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO клиппинг
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = F.mse_loss(values.squeeze(), returns)
            
            # Общий лосс
            total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy.mean()
            
            # Обновляем параметры
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        # Очищаем буферы
        self.clear_buffer()
    
    def clear_buffer(self):
        """Очистить буферы опыта"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def save(self, filename):
        """Сохранить модель"""
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filename)
    
    def load(self, filename):
        """Загрузить модель"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])