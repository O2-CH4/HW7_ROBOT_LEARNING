import numpy as np
from gymnasium import ActionWrapper
from gymnasium.core import ActType, WrapperActType

class GaussianActionWrapper(ActionWrapper):
    def __init__(self, env, scale):
        super().__init__(env)
        self.noise_scale = scale

    def action(self, action: WrapperActType) -> ActType:
        noisy_action = np.random.normal(loc=action, scale=self.noise_scale, size=action.shape)
        return np.clip(noisy_action, self.action_space.low, self.action_space.high)
