import numpy as np

from locobotSim.env.locobot_env import LocobotEnv
from locobotSim.planners.rrt import RRT

env = LocobotEnv(seed=42, num_humans=0)
obs = env.get_obs()

rrt = RRT(env=env, start=obs[:3], goal=env.sites["core"])

rrt.run()
rrt.visualize()



