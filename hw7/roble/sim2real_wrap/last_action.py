import numpy as np
from gymnasium import ActionWrapper
from gymnasium.core import ActType

class LastActionWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_action = None

    def action(self, action: ActType) -> ActType:
        self.last_action = action
        return action