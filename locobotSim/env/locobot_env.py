import os, random

import numpy as np
import mujoco as mj

from locobotSim import utils
from locobotSim.env.humans import Humans
from locobotSim.env import in_env_collision, convert_robot_orientation_to_quat

np.set_printoptions(suppress=True)


class LocobotEnv:
    max_speed = 70
    LOCOBOT_RADIUS = 0.2

    sites = {
        "werblin": np.array((-2.5, 9.5)),
        "core": np.array((2.75, 9.5)),
        "bsc": np.array((-2.5, -3.0)),
        "ee": np.array((-2.5, -10)),
        "arc": np.array((0.25, -5.5)),
    }

    site_indices = {
        "werblin": 0,
        "core": 1,
        "bsc": 2,
        "ee": 3,
        "arc": 4,
    }

    indices_to_sites = {
        0: "werblin",
        1: "core",
        2: "bsc",
        3: "ee",
        4: "arc",
    }

    def __init__(self, num_humans=5, seed=None):
        self.num_humans = num_humans
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.model = mj.MjModel.from_xml_path(
            os.path.join(os.path.dirname(__file__), "../assets/world.xml")
        )
        self.data = mj.MjData(self.model)

        self.set_robot_pose(np.array([0, 0]), 1.57, 0.0)

        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)

        self.humans = Humans(
            num_humans,
            self.model,
            self.data,
            self.sites,
            self.get_robot_position,
        )

        self.reset()
        self.initialize()

    def get_obs(self):
        position = self.get_robot_position()
        velocity = self.get_robot_velocity()
        orientation = self.get_robot_orientation()
        human_positions = np.zeros(10)
        human_positions[:2 * self.num_humans] = self.humans.get_human_positions().reshape((1, -1))[0]
        human_goals = self.humans.goals
        human_goals = [self.site_indices[goal] for goal in human_goals]

        obs = np.concatenate(
            [
                position,
                [
                    orientation,
                    velocity,
                ],
                human_positions,
                human_goals,
            ]
        )

        return obs

    def convert_obs_to_state(self, obs):
        position = obs[:2]
        orientation = obs[2]
        velocity = obs[3]
        human_positions = obs[4:4 + (self.num_humans * 2)].reshape((-1, 2))
        human_goal_site_idx = obs[14:]
        human_goal_sites = [self.indices_to_sites[goal] for goal in human_goal_site_idx]

        return position, orientation, velocity, human_positions, human_goal_sites

    def reset(self, init_obs=None, init_robot_pose=None):
        if init_robot_pose is not None and init_obs is not None:
            raise ValueError("Cannot specify both init_robot_pose and init_obs")

        self.data.qpos = np.copy(self.initial_qpos)
        self.data.qvel = np.copy(self.initial_qvel)
        self.data.ctrl = np.zeros(self.model.nu)
        self.data.time = 0

        if init_obs is not None:
            position, orientation, velocity, human_positions, human_goals = self.convert_obs_to_state(init_obs)

            self.set_robot_pose(position, orientation, velocity)
            self.humans.reset(starts=human_positions, goals=human_goals)
        else:
            self.humans.reset()

        if init_robot_pose is not None:
            position, orientation, velocity = init_robot_pose

            self.set_robot_pose(position, orientation, velocity)

    def set_robot_pose(self, position, orientation, velocity=0.0):
        # Swap x and y
        self.data.qpos[:2] = position
        self.data.qpos[3] = orientation
        self.data.qvel[0] = velocity

    def get_robot_position(self):
        position = self.data.qpos[:2]
        return position

    def get_robot_orientation_quat(self):
        return self.data.xquat[1]

    def get_robot_orientation(self):
        return utils.convert_quat_to_euler(self.get_robot_orientation_quat())[2]

    def get_robot_velocity(self):
        return self.data.qvel[0]

    def apply_action(self, action):
        self.data.ctrl[0] = action[0] * self.max_speed
        self.data.ctrl[1] = action[1] * self.max_speed

    def step(self, action=None):
        self.humans.step()
        if action is not None:
            self.apply_action(action)
            if self.data.qpos[0] < -np.pi:
                self.data.qpos[0] += 2* np.pi
            if self.data.qpos[0] > np.pi:
                self.data.qpos[0] -= 2* np.pi

        mj.mj_step(self.model, self.data)

    def initialize(self):
        action = [0, 0]
        self.apply_action(action)
        for _ in range(300):
            for i in range(self.model.nv):
                self.data.qacc_warmstart[i] = 0
            mj.mj_step(self.model, self.data)

    def render(self):
        mj.mj_render(self.model, self.data, mj.MJCAT_ALL, mj.MJRF_CAMERA)

    def forward_prop(self, config, u, prop_steps):
        self.reset(init_obs=config)
        traj = [config]
        is_valid = True
        for _ in range(prop_steps*10):
            self.step(u)
            obs = self.get_obs()
            traj.append(obs)

            if in_env_collision(obs[:2]) or self.humans.in_agent_collision(obs[:2], exclude_robot=True):
                is_valid = False
                break

        return traj, is_valid