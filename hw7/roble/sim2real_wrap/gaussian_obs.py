import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.core import ObsType, WrapperObsType

class GaussianObservationWrapper(ObservationWrapper):
    def __init__(self, env, scale):
        super().__init__(env)
        self.noise_scale = scale

    def observation(self, observation: ObsType) -> WrapperObsType:
        noisy_observation = np.random.normal(observation, scale=self.noise_scale, size=observation.shape)
        return np.clip(noisy_observation, self.observation_space.low, self.observation_space.high)