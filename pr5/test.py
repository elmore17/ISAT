from stable_baselines3 import PPO
import gym
from gym import spaces
import numpy as np

class LiquidDistributionEnv(gym.Env):
    def __init__(self):
        super(LiquidDistributionEnv, self).__init__()
        self.action_space = spaces.MultiDiscrete([5] * 5)  # 5 стаканов
        self.observation_space = spaces.Box(low=0, high=1000, shape=(5,), dtype=np.float32)
        self.state = np.random.randint(0, 1000, size=(5,))
    
    def reset(self, initial_state=None):
        if initial_state is not None:
            self.state = np.array(initial_state).astype(np.float32)
        else:
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
        else:
            transfer_amount = 0  # Если в стакане нет жидкости, ничего не переливаем
        
        done = np.all(self.state == np.mean(self.state))
        reward = -np.sum(np.abs(self.state - np.mean(self.state)))  # Награда за равенство
        
        return self.state, reward, done, {"from_idx": from_idx, "to_idx": to_idx, "transfer_amount": transfer_amount}
    
    def render(self):
        print(f"Стаканы: {self.state}")

# Загрузка модели
model = PPO.load("pr5/liquid_distribution_model")

# Тестирование модели на заданном состоянии стаканов
def test_model(model, initial_state):
    env_test = LiquidDistributionEnv()
    env_test.reset(initial_state)  # Устанавливаем начальное состояние
    
    done = False
    total_reward = 0
    steps = 0  # Счетчик шагов
    states_history = []  # Список для хранения состояний на каждом шаге
    
    print("Начальное состояние стаканов:", initial_state)
    
    while not done:
        action, _states = model.predict(env_test.state)  # Предсказание действия
        states_history.append(env_test.state.copy())  # Сохраняем текущее состояние
        
        # Выполнение действия в среде и получение информации о переливании
        state, reward, done, info = env_test.step(action)
        
        total_reward += reward  # Обновляем общую награду
        
        # Выводим информацию о действии
        if info['transfer_amount'] < 0:
            print(f"Шаг {steps + 1}: Перелито {abs(info['transfer_amount'])} из стакана {info['to_idx'] + 1} в стакан {info['from_idx'] + 1}")
        else:
            print(f"Шаг {steps + 1}: Перелито {abs(info['transfer_amount'])} из стакана {info['from_idx'] + 1} в стакан {info['to_idx'] + 1}")
        
        # Проверка на завершение эпизода
        steps += 1  # Увеличиваем счетчик шагов

    # Выводим все состояния за шаги
    for i, state in enumerate(states_history):
        print(f"Состояние на шаге {i + 1}: {state}")

    print("Конечное состояние стаканов:", env_test.state)
    print("Общая награда:", total_reward)
    print("Количество итераций:", steps)

# Задаем начальное состояние стаканов и тестируем модель
initial_state = [28, 58, 8, 58, 78]
test_model(model, initial_state)