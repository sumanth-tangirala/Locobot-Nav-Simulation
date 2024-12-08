import numpy as np

from locobotSim.env.locobot_env import LocobotEnv
from locobotSim.planners.rrt import RRT

results = {}
for humans in range(6):
    env = LocobotEnv(seed=42, num_humans=humans, human_agent_planner="RVO")
    obs = env.get_obs()

    with open("rrt_results.txt", "a") as f:
        f.write(f"Number of humans: {humans}\n")

    human_results = {}
    for site in env.sites.keys():
        site_total_steps = []
        site_total_nodes = []
        failures = 0
        for i in range(10):
            env.reset()
            env.initialize()
            rrt = RRT(env=env, start=obs[:3], goal=env.sites[site])
            _, total_steps, total_nodes = rrt.run()
            rrt.visualize(fname=f"_{site}_{humans}_{i}")

            if total_steps is None:
                failures += 1
            else:
                site_total_steps.append(total_steps)

            site_total_nodes.append(total_nodes)

        human_results[site] = (site_total_steps, failures, site_total_nodes)

        with open("rrt_results.txt", "a") as f:
            f.write(f"Site: {site}\n")
            f.write(f"Total steps: {site_total_steps}, ")
            f.write(f"Average steps: {np.mean(site_total_steps)}, ")
            f.write(f"Total Nodes: {site_total_nodes}, ")
            f.write(f"Failures: {failures}")
            f.write("\n")

    results[humans] = human_results









