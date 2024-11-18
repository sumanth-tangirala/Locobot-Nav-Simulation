import gym
import numpy as np
import mujoco as mj

from gym import spaces

from locobotSim import utils
from locobotSim.locobot_env import Locobot

np.set_printoptions(suppress=True)


class LocobotTrainingEnv(gym.Env):
    goal_limit = 1
    distance_threshold = 0.5

    def __init__(
        self,
        max_steps=30,
        noisy=False,
        noise_scale=0.01,
        return_full_trajectory=False,
        prop_steps=100,
        max_speed=700,
    ):

        print("Environment Configuration: ")
        print("Max Steps: ", max_steps)
        print("Prop Steps: ", prop_steps)

        self.locobot = Locobot(max_speed)

        self.locobot.reset()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        self.obs_dims = 4  # x, y, theta, v
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
        self.noisy = noisy
        self.noise_scale = noise_scale

    def reset(self, seed=None, options=dict()):
        self.locobot.reset()
        self.steps = 0

        goal = options.get("goal", None)

        if goal is None:
            self.goal = np.random.uniform(
                -self.goal_limit,
                self.goal_limit,
                size=(self.goal_dims,),
            )
        else:
            self.goal = goal
        return self._get_obs()

    def _get_obs(self):
        obs = self.locobot.get_obs()

        if self.noisy:
            obs += np.random.normal(0, self.noise_scale, obs.shape)

        achieved_goal = np.array([obs[0], obs[1]])

        return {
            "observation": np.float32(obs),
            "achieved_goal": np.float32(achieved_goal),
            "desired_goal": np.float32(self.goal),
        }

    def _terminal(self, s, g):
        return utils.pos_distance(s, g) < self.distance_threshold

    def compute_reward(self, ag, dg, info):
        return -(utils.pos_distance(ag, dg) >= self.distance_threshold).astype(
            np.float32
        )

    def step(self, action):
        self.steps += 1

        self.locobot.apply_action(action)

        current_traj = []
        for _ in range(self.prop_steps):
            for i in range(self.locobot.model.nv):
                self.locobot.data.qacc_warmstart[i] = 0

            self.locobot.step()
            if self.return_full_trajectory:
                current_traj.append(self._get_obs()["achieved_goal"])

        obs = self._get_obs()
        info = {
            "is_success": self._terminal(obs["achieved_goal"], obs["desired_goal"]),
            "traj": np.array(current_traj),
        }
        done = (
            self._terminal(obs["achieved_goal"], obs["desired_goal"])
            or self.steps >= self.max_steps
        )
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {})
        return obs, reward, done, info


if __name__ == "__main__":
    env = LocobotTrainingEnv(max_steps=10, prop_steps=100)
    obs = env.reset()
    traj = [np.copy(obs["observation"])]

    for _ in range(100):
        next_action = env.action_space.sample()
        next_action = np.array([1.0, 1.0])
        obs, reward, done, _ = env.step(next_action)
        traj.append(np.copy(obs["observation"]))

    traj = np.array(traj)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    traj = np.vstack(traj)
    plt.plot(traj[:, 0], traj[:, 1], "r-")
    plt.savefig("env_test.png")
