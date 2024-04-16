import numpy as np
from gymnasium import Wrapper
from typing import Tuple

class NoOpResetWrapper(Wrapper):
    def __init__(self, env, max_num_random_act=4):
        super().__init__(env)
        self.max_no_op_actions = max_num_random_act

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        observation, original_info = super().reset(**kwargs)
        num_no_op = np.random.randint(0, self.max_no_op_action+1)
        for _ in range(num_no_op):
            action = np.zeros(self.action_space.shape)
            observation, r, done, trunc, info = super().step(action)
        return observation, original_info