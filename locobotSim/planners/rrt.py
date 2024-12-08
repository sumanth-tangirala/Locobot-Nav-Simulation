import random

from tqdm import tqdm
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt

from locobotSim.env.locobot_env import LocobotEnv
from locobotSim.env import get_random_pose, ENV_X_MAX, ENV_X_MIN, ENV_Y_MAX, ENV_Y_MIN, in_env_collision
from locobotSim.utils.rrt_utils import RRTNode, compute_configs_distance
from locobotSim.constants import (
    GOAL_POS_DIST,
    MIN_PROP_STEPS,
    MAX_PROP_STEPS,
    GOAL_BIAS_PROB,
    MAXIMUM_PROP_COUNT,
)


class RRT:
    trajectory_found = False
    trajectory = []
    controls = []
    prop_steps = []

    def __init__(self, start, goal, env: LocobotEnv):
        self.start = start
        self.goal = np.array([*goal, 0])
        self.env = env

        obs = self.env.get_obs()
        obs[:2] = start[:2]

        if in_env_collision(obs[:2]) or self.env.humans.in_agent_collision(obs[:2], exclude_robot=True):
            raise ValueError("Start in collision")

        self.tree = [RRTNode(obs)]
        self.tree_nodes = np.array([obs[:3]])

        print('Starting RRT with:')
        print('Start:', start)
        print('Goal:', goal)

    def _should_terminate(self, config):
        return np.linalg.norm(config[:2] - self.goal[:2]) < GOAL_POS_DIST

    def _propagate(self, random_node, u=None):
        if u is None:
            u0 = np.random.uniform(-1, 1)
            u1 = np.random.uniform(-1, 1)

            u = np.array([u0, u1])

        prop_steps = random.randint(MIN_PROP_STEPS, MAX_PROP_STEPS)

        traj, is_valid = self.env.forward_prop(random_node.config.copy(), u, prop_steps)

        return traj[-1], u, prop_steps, is_valid, traj

    def run(self):
        min_dist = np.inf
        goal_node = None

        out_of_bounds_count = 0
        with tqdm(total= MAXIMUM_PROP_COUNT) as pbar:
            while len(self.tree) < MAXIMUM_PROP_COUNT:
                if np.random.uniform(0, 1) < GOAL_BIAS_PROB:
                    random_pose = self.goal
                else:
                    random_pose = get_random_pose()

                node_distances = compute_configs_distance(self.tree_nodes, random_pose)
                min_node_idx = np.argmin(node_distances)
                random_node = self.tree[min_node_idx]

                new_config, control, prop_steps, is_valid, traj = self._propagate(random_node)

                # new_x, new_y = new_config[:2]
                # if np.abs(new_x) > 49 or np.abs(new_y) > 49:
                #     out_of_bounds_count += 1
                #     if out_of_bounds_count > 100:
                #         print("Out of bounds")
                #         return None
                #     continue



                if new_config is None or not is_valid:
                    if new_config is not None:
                        new_node = RRTNode(new_config, parent=random_node, is_valid=is_valid)
                    continue

                new_node = RRTNode(new_config, parent=random_node, is_valid=is_valid)
                goal_dist = new_node.compute_distance(self.goal, individual=False)

                if goal_dist < min_dist:
                    min_dist = goal_dist
                    print('Reached closer to goal:', new_config[:2], 'Distance:', min_dist)

                random_node.add_child(new_node, control, prop_steps)

                self.tree.append(new_node)
                self.tree_nodes = np.vstack([self.tree_nodes, new_config[:3]])
                dist = new_node.compute_distance(self.goal, individual=False)

                min_dist = min(min_dist, dist)

                if self._should_terminate(new_config):
                    goal_node = new_node
                    break

                pbar.update(1)

        if goal_node is None:
            node_distances = compute_configs_distance(self.tree_nodes, self.goal)
            min_node_idx = np.argmin(node_distances)
            closest_node = self.tree[min_node_idx]
            print("Could not reach goal")
            print("Closest node to goal: ", closest_node.config)
            return None

        self.trajectory_found = True

        reverse_traj = []
        reverse_controls = []
        reverse_prop_steps = []

        node = goal_node

        while node is not None:
            if node.parent:
                control, prop_steps = node.parent.children[node]
                reverse_controls.append(control)
                reverse_prop_steps.append(prop_steps)

            reverse_traj.append(node.config.copy())

            node = node.parent

        assert len(reverse_traj) == len(reverse_controls) + 1
        assert len(reverse_traj) == len(reverse_prop_steps) + 1

        self.trajectory = list(reversed(reverse_traj))
        self.controls = list(reversed(reverse_controls))
        self.prop_steps = list(reversed(reverse_prop_steps))

        return self.trajectory

    def visualize_rrt(self):
        print("Generating Animation...")
        fig, ax = plt.subplots()
        ax.set_xlim(ENV_X_MIN, ENV_X_MAX)
        ax.set_ylim(ENV_Y_MIN, ENV_Y_MAX)
        ax.set_aspect("equal")

        # Draw static elements
        ax.add_patch(
            plt.Circle(  # type: ignore
                (self.goal[0], self.goal[1]), GOAL_POS_DIST, color="red", fill=False
            )
        )
        ax.plot(self.start[0], self.start[1], "bo")

        # Pre-compute edges
        edges = []

        for node in self.tree:
            if node.parent is not None:
                edges.append(
                    [
                        (node.parent.config[0], node.parent.config[1]),
                        (node.config[0], node.config[1]),
                    ]
                )

        line_segments = LineCollection(edges, colors="blue", linewidths=0.5)
        ax.add_collection(line_segments)  # type: ignore

        def update(frame):
            # Only modify what needs to be changed
            line_segments.set_segments(edges[:frame])
            return (line_segments,)

        anim = FuncAnimation(
            fig, update, frames=len(self.tree) + 1, interval=30, blit=True
        )

        print("Saving Animation...")
        file_name = "./data/rrt/tree."

        anim.save(file_name + "mp4", writer="ffmpeg")

    def visualize_path(self):
        plt.figure(figsize=(8, 8))

        if not self.trajectory_found:
            print("No trajectory found")
            return

        for node in self.trajectory:
            config = node.config
            plt.plot(config[0], config[1], "bo")


    def visualize(self):
        self.visualize_rrt()
        self.visualize_path()
