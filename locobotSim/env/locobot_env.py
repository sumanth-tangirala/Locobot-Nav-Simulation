import os

import numpy as np
import mujoco as mj

from locobotSim import utils
from locobotSim.env.humans import Humans

np.set_printoptions(suppress=True)


class LocobotEnv:
    max_speed = 1000
    LOCOBOT_RADIUS = 0.2

    sites = {
        "werblin": np.array((-2.5, 9.5)),
        "core": np.array((2.75, 9.5)),
        "bsc": np.array((-2.5, -3.0)),
        "ee": np.array((-2.5, -10)),
        "arc": np.array((0.25, -5.5)),
    }

    def __init__(self, num_humans=5):
        self.model = mj.MjModel.from_xml_path(
            os.path.join(os.path.dirname(__file__), "../assets/world.xml")
        )
        self.data = mj.MjData(self.model)

        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)
        self.humans = Humans(
            num_humans, self.model, self.data, self.sites, self.get_robot_position
        )

        self.reset()
        self.initialize()

    def reset(self, init_obs=None, init_robot_pose=None):
        if init_robot_pose is not None and init_obs is not None:
            raise ValueError("Cannot specify both init_robot_pose and init_obs")

        self.data.qpos = np.copy(self.initial_qpos)
        self.data.qvel = np.copy(self.initial_qvel)
        self.data.ctrl = np.zeros(self.model.nu)
        self.data.time = 0
        self.humans.reset()

        if init_obs is not None:
            position, orientation, velocity, human_positions = (
                self.convert_obs_to_state(init_obs)
            )

            self.data.qpos[0][:2] = position
            self.data.qpos[0][3:7] = orientation
            self.data.qvel[0][0] = velocity
            self.humans.set_human_positions(human_positions)

        if init_robot_pose is not None:
            position, orientation, velocity = init_robot_pose

            self.data.qpos[0][:2] = position
            self.data.qpos[0][3:7] = orientation

    def get_robot_position(self):
        return self.data.xpos[1][:2]

    def get_obs(self):
        position = self.get_robot_position()
        velocity = self.data.qvel[0]
        orientation_quat = self.data.xquat[1]
        orientation = utils.convert_quat_to_euler(orientation_quat)[2]
        human_positions = self.humans.get_human_positions().reshape((1, -1))[0]

        obs = np.concatenate(
            [
                position,
                [
                    orientation,
                    velocity,
                ],
                human_positions,
            ]
        )

        return obs

    def convert_obs_to_state(self, obs):
        position = obs[:2]
        orientation_euler = obs[2]
        velocity = obs[3]
        human_positions = obs[4:].reshape((-1, 2))

        orientation = utils.convert_euler_to_quat(np.array([0, 0, orientation_euler]))

        return position, orientation, velocity, human_positions

    def apply_action(self, action):
        self.data.ctrl[0] = action[0] * self.max_speed
        self.data.ctrl[1] = action[1] * self.max_speed

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


    def forward_prop(config, u, prop_steps):
        traj = [config]

        for _ in range(prop_steps):
            pass

        return traj