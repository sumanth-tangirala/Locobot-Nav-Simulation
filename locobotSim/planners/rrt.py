import random
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt

from locobotSim.utils.rrt_utils import RRTNode
from locobotSim.constants import (
    GOAL_POS_DIST,
    MIN_PROP_STEPS,
    MAX_PROP_STEPS,
    GOAL_BIAS_PROB,
    MAXIMUM_PROP_COUNT,
)


class RRT:
    visualization_directory = "rrt"
    trajectory_found = False
    trajectory = []
    controls = []
    prop_steps = []

    def __init__(self, start, goal):
        self.start = start
        self.goal = goal

        self.tree = [RRTNode(start)]

    def _should_terminate(self, config):
        return np.linalg.norm(config[:2] - self.goal[:2]) < GOAL_POS_DIST

    def _propagate(self, random_node):
        u0 = np.random.uniform(-1, 1)
        u1 = np.random.uniform(-1, 1)

        u = np.array([u0, u1])

        prop_steps = random.randint(MIN_PROP_STEPS, MAX_PROP_STEPS)

        traj = (random_node.config.copy(), u, prop_steps)

        return traj[-1], u, prop_steps

    def run(self):
        min_dist = np.inf
        goal_node = None

        out_of_bounds_count = 0

        while len(self.tree) < MAXIMUM_PROP_COUNT:
            if np.random.uniform(0, 1) < GOAL_BIAS_PROB:
                random_config = self.goal
            else:
                random_config = generate_random_config()

            min_random_dist = np.inf

            for node in self.tree:
                dist = node.compute_distance(random_config)

                if min_random_dist > dist:  # type: ignore
                    min_random_dist = dist

                    random_node = node

            new_config, control, prop_steps = self._propagate(random_node)

            new_x, new_y = new_config[:2]
            # breakpoint()
            if np.abs(new_x) > 49 or np.abs(new_y) > 49:
                out_of_bounds_count += 1
                if out_of_bounds_count > 100:
                    print("Out of bounds")
                    return None
                continue

            if new_config is None:
                continue

            new_node = RRTNode(new_config, parent=random_node)

            random_node.add_child(new_node, control, prop_steps)

            self.tree.append(new_node)

            dist = new_node.compute_distance(self.goal, individual=False)

            # pos_dist, ang_dist = new_node.compute_distance(
            #     self.goal, individual=True)
            # if min_dist > dist:
            #     print(min_dist, pos_dist, ang_dist, len(self.tree))

            min_dist = min(min_dist, dist)

            if self._should_terminate(new_config):
                goal_node = new_node
                break

        if goal_node is None:
            print("Could not reach goal")
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
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
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
        file_name = "./visualizer/out/rrt/tree."

        anim.save(file_name + "mp4", writer="ffmpeg")

    def visualize_path(self):
        plt.figure(figsize=(8, 8))

        for node in self.trajectory:
            config = node.config
            plt.plot(config[0], config[1], "bo")
