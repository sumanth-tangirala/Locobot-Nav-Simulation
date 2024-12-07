from locobotSim.env.training_env import LocobotTrainingEnv
from gym.envs.registration import register
import numpy as np


class EnvironmentFactory:
    def __init__(self, return_full_trajectory=False, prop_steps=5, max_steps=250):
        self.return_full_trajectory = return_full_trajectory
        self.prop_steps = prop_steps
        self.max_steps = max_steps

    def register_environments(self):
        register(id="LocobotEnv-v0", entry_point=LocobotTrainingEnv,
                 kwargs={'return_full_trajectory': self.return_full_trajectory, 'prop_steps': self.prop_steps,
                         'max_steps': self.max_steps})

    def get_applied_action(self, action):
        return action
