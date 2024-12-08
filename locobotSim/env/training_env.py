import gym
import numpy as np
import mujoco as mj

from gym import spaces

from locobotSim import utils
from locobotSim.env.locobot_env import LocobotEnv

np.set_printoptions(suppress=True)


class LocobotTrainingEnv(gym.Env):
    goal_limit = 1
    distance_threshold = 0.5

    def __init__(
        self,
        max_steps=100,
        return_full_trajectory=False,
        prop_steps=5,
        num_humans=5,
    ):

        print("Environment Configuration: ")
        print("Max Steps: ", max_steps)
        print("Prop Steps: ", prop_steps)
        print("Number of Humans: ", num_humans)

        self.locobot = LocobotEnv(num_humans=num_humans)
        self.sites = list(self.locobot.sites.values())
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        self.obs_dims = 4 + 5 * 2  # x, y, theta, v, (human_x, human_y) * 5
        self.goal_dims = 2  # x, y

        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.obs_dims,)
                ),
                "achieved_goal": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.goal_dims,)
                ),
                "desired_goal": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.goal_dims,)
                ),
            }
        )

        self.max_steps = max_steps
        self.return_full_trajectory = return_full_trajectory
        self.prop_steps = prop_steps

        self.reset()

    def reset(self, init_obs=None, goal=None):
        self.locobot.reset(init_obs)
        self.steps = 0

        if goal is None:
            self.goal = self.sites[np.random.randint(len(self.sites))]
        else:
            self.goal = goal
        return self._get_obs()

    def _get_obs(self):
        obs = self.locobot.get_obs()
        obs = obs[: self.obs_dims]

        achieved_goal = np.array([obs[0], obs[1]])

        return {
            "observation": np.float32(obs),
            "achieved_goal": np.float32(achieved_goal),
            "desired_goal": np.float32(self.goal),
        }

    def in_collision_with_human(self, robot_position):
        human_positions = self.locobot.humans.get_human_positions()
        human_dist = np.linalg.norm(human_positions - robot_position, axis=1)
        return np.any(
            human_dist
            < (self.locobot.humans.HUMAN_RADIUS + self.locobot.LOCOBOT_RADIUS)
        )

    def _terminal(self, s, g, in_collision_with_human):
        return (
            in_collision_with_human
            or utils.pos_distance(s, g) < self.distance_threshold
        )

    def compute_reward(self, ag, dg, info):
        return -(utils.pos_distance(ag, dg) >= self.distance_threshold).astype(
            np.float32
        )

    def step(self, action):
        self.steps += 1

        self.locobot.apply_action(action)

        current_traj = []
        in_collision_with_human = False
        for _ in range(self.prop_steps):
            for i in range(self.locobot.model.nv):
                self.locobot.data.qacc_warmstart[i] = 0

            self.locobot.step()
            in_collision_with_human = self.in_collision_with_human(
                self.locobot.get_robot_position()
            )
            if self.return_full_trajectory:
                current_traj.append(self._get_obs()["achieved_goal"])

            if in_collision_with_human:
                break

        obs = self._get_obs()
        is_terminal = self._terminal(
            obs["achieved_goal"], obs["desired_goal"], in_collision_with_human
        )

        is_success = False

        if is_terminal and not in_collision_with_human:
            is_success = True

        info = {
            "is_success": is_success,
            "traj": np.array(current_traj),
            "in_collision_with_human": in_collision_with_human,
        }
        done = is_success or self.steps >= self.max_steps
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {})
        return obs, reward, done, info


if __name__ == "__main__":
    env = LocobotTrainingEnv(max_steps=10, prop_steps=100)
    obs = env.reset()
    traj = [np.copy(obs["observation"])]

    for _ in range(100):
        next_action = env.action_space.sample()
        obs, reward, done, info = env.step(next_action)
        traj.append(np.copy(obs["observation"]))

    print(info)

    traj = np.array(traj)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    traj = np.vstack(traj)
    plt.plot(traj[:, 0], traj[:, 1], "r-")
    plt.savefig("env_test.png")
