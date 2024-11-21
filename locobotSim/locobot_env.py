import os

import numpy as np
import mujoco as mj

from locobotSim import utils
from locobotSim.humans import Humans

np.set_printoptions(suppress=True)


class Locobot:
    max_speed = 1000

    sites = {
        "werblin": np.array((-2.5, 9.5)),
        "core": np.array((2.75, 9.5)),
        "bsc": np.array((-2.5, -3.0)),
        "ee": np.array((-1.5, -9.5)),
    }

    def __init__(self, num_humans=0):
        self.model = mj.MjModel.from_xml_path(
            os.path.join(os.path.dirname(__file__), "assets/world.xml")
        )
        self.data = mj.MjData(self.model)

        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)
        self.humans = Humans(
            num_humans, self.model, self.data, self.sites, self.get_robot_position
        )

    def reset(self, human_starts=None, human_goals=None):
        self.data.qpos = np.copy(self.initial_qpos)
        self.data.qvel = np.copy(self.initial_qvel)
        self.data.ctrl = np.zeros(self.model.nu)
        self.data.time = 0

        self.humans.reset(starts=human_starts, goals=human_goals)

        mj.mj_forward(self.model, self.data)

        self.initialize()

    def get_robot_position(self):
        return self.data.xpos[1][:2]

    def get_obs(self):
        position = self.get_robot_position()
        velocity = self.data.qvel[0]
        orientation = utils.convert_quat_to_euler(self.data.xquat[1])[2]
        obs = np.concatenate([position, [orientation, velocity]])

        return obs

    def apply_action(self, action):
        self.data.ctrl[0] = action[0] * self.max_speed
        self.data.ctrl[1] = action[1] * self.max_speed

        print(self.data.ctrl)

    def step(self):
        self.humans.step()
        mj.mj_step(self.model, self.data)

    def initialize(self):
        action = [0, 0]
        self.apply_action(action)
        for _ in range(300):
            for i in range(self.model.nv):
                self.data.qacc_warmstart[i] = 0
            mj.mj_step(self.model, self.data)
