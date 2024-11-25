import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from tqdm import tqdm

# Создание среды
class LiquidDistributionEnv(gym.Env):
    def __init__(self):
        super(LiquidDistributionEnv, self).__init__()
        self.action_space = spaces.MultiDiscrete([5] * 5)  # 5 стаканов
        self.observation_space = spaces.Box(low=0, high=1000, shape=(5,), dtype=np.float32)
        self.state = np.random.randint(0, 1000, size=(5,))
    
    def reset(self):
        self.state = np.random.randint(0, 1000, size=(5,))
        return self.state
    
    def step(self, action):
        # Переливание жидкости в соответствии с действием
        from_idx = action[0]
        to_idx = action[1]
        
        if self.state[from_idx] > 0:
            transfer_amount = min(self.state[from_idx], (np.mean(self.state) - self.state[to_idx]))
            self.state[from_idx] -= transfer_amount
            self.state[to_idx] += transfer_amount
        
        done = np.all(self.state == np.mean(self.state))
        reward = -np.sum(np.abs(self.state - np.mean(self.state)))  # Награда за равенство
        
        return self.state, reward, done, {}
    
    def render(self):
        print(f"Стаканы: {self.state}")

# Обучение агента
env = LiquidDistributionEnv()
model = PPO("MlpPolicy", env, verbose=1)

# Список для хранения наград
rewards = []

# Обучение модели с индикатором прогресса
num_episodes = 500  # Количество эпизодов

for i in tqdm(range(num_episodes), desc="Обучение агента", unit="эпизод"):
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    
    rewards.append(total_reward)
    model.learn(total_timesteps=100)

# Сохранение модели
model.save("liquid_distribution_model")

# Сохранение кривой обучения
plt.plot(rewards)
plt.title('Кривая обучения агента')
plt.xlabel('Эпизоды')
plt.ylabel('Награда')
plt.savefig('training_curve.jpg')
plt.show()