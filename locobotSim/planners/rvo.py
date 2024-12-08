import numpy as np

from locobotSim.env.locobot_env import LocobotEnv


class RVO:
    trajectory = []

    def __init__(self, start, goal, env: LocobotEnv):
        self.start = start
        self.goal = np.array([*goal, 0])
        self.env = env


        print('Starting RVO with:')
        print('Start:', start)
        print('Goal:', goal)