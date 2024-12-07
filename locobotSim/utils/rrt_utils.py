import numpy as np
from locobotSim.constants import ALPHA
from locobotSim.utils import angular_diff


class RRTNode:
    def __init__(self, config, parent=None):
        self.config = config
        self.position = config[:2]
        self.orientation = config[2]
        self.human_positions = config[3:]
        self.parent: RRTNode = parent  # type: ignore

        self.children = {}

    def __str__(self) -> str:
        return f"{self.config}"

    def __repr__(self) -> str:
        return f"{self.config}"

    def add_child(self, child, action, prop_steps):
        self.children[child] = [action, prop_steps]

    def compute_distance(self, other_config, individual=False):
        pos_dist = float(np.linalg.norm(self.position - other_config[:2]))
        angle_diff = float(np.abs(angular_diff(self.orientation, other_config[2])))

        if individual:
            return pos_dist, angle_diff

        return (ALPHA * pos_dist) + ((1 - ALPHA) * angle_diff)
