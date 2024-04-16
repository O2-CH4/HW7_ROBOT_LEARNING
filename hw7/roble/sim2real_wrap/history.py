import collections
import gymnasium
import numpy as np
from gymnasium import Wrapper

class HistoryWrapper(Wrapper):
    def __init__(self, env, length):
        super().__init__(env)
        self.history_length = length

        low, high = env.observation_space.low, env.observation_space.high
        low = np.tile(low, (length, 1)).flatten()
        high = np.tile(high, (length, 1)).flatten()

        self.observation_space = gymnasium.spaces.Box(low=low, high=high)

    def reset(self):
        self.history = collections.deque(maxlen=self.history_length)
        observation = super().reset()
        for _ in range(self.history_length):
            self.history.append(observation)
        return np.array(self.history).flatten()

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self.history.append(observation)
        return np.array(self.history).flatten(), reward, done, info