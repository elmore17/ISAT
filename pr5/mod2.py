import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from tqdm import tqdm

# Среда
class LiquidDistributionEnv(gym.Env):
    def __init__(self):
        super(LiquidDistributionEnv, self).__init__()
        self.action_space = spaces.MultiDiscrete([5] * 5)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(5,), dtype=np.float32)
        self.state = np.random.randint(0, 1000, size=(5,))
    
    def reset(self):
        self.state = np.random.randint(0, 1000, size=(5,))
        return self.state
    
    def step(self, action):
        from_idx = action[0]
        to_idx = action[1]
        
        if self.state[from_idx] > 0:
            transfer_amount = min(self.state[from_idx], (np.mean(self.state) - self.state[to_idx]))
            self.state[from_idx] -= transfer_amount
            self.state[to_idx] += transfer_amount
        
        done = np.all(self.state == np.mean(self.state))
        reward = -np.sum(np.abs(self.state - np.mean(self.state)))
        return self.state, reward, done, {}
    
    def render(self):
        print(f"Стаканы: {self.state}")

# Обучение
env = LiquidDistributionEnv()
model = PPO("MlpPolicy", env, verbose=1)

# Список наград
rewards = []

# Обучение с отслеживанием наград
num_episodes = 500  # Общее количество эпизодов
interval = 8       # Интервал для вычисления средней награды
reward_intervals = []

for i in tqdm(range(num_episodes), desc="Обучение агента", unit="эпизод"):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    
    rewards.append(total_reward)

    # Сохраняем среднюю награду каждые `interval` эпизодов
    if (i + 1) % interval == 0:
        avg_reward = np.mean(rewards[-interval:])  # Средняя награда за интервал
        reward_intervals.append(avg_reward)
        print(f"Средняя награда за последние {interval} эпизодов: {avg_reward}")
    
    # Продолжаем обучение модели
    model.learn(total_timesteps=500)

# Построение точечного графика
x_values = list(range(interval, num_episodes + 1, interval))
plt.scatter(x_values, reward_intervals, color='blue')
plt.title('Кривая обучения агента (точечный граф)')
plt.xlabel('Эпизоды')
plt.ylabel('Средняя награда за интервал')
plt.grid(True)
plt.savefig('training_curve_points_fixed.jpg')
plt.show()
