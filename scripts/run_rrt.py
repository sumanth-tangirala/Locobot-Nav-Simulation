import numpy as np

from locobotSim.env.locobot_env import LocobotEnv
from locobotSim.planners.rrt import RRT

env = LocobotEnv(seed=42, num_humans=1)
obs = env.get_obs()

rrt = RRT(env=env, start=obs[:3], goal=env.sites["werblin"])

rrt.run()
rrt.visualize()



