import numpy as np
from gymnasium import Wrapper

class ActionRepeatWrapper(Wrapper):
    def __init__(self, env, max_repeats):
        super().__init__(env)
        self.max_repeats = max_repeats

    def step(self, action):
        total_reward = 0
        last_observation = None
        done = False
        trunc = False
        info = {}
        for _ in range(np.random.randint(1, self.max_repeats + 1)):
            last_observation, reward, done, trunc, info = super().step(action)
            total_reward += reward
            if done or trunc:
                break
        return last_observation, total_reward, done, trunc, info