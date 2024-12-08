import numpy as np

from locobotSim import utils

ENV_X_MIN = -2.8
ENV_X_MAX = 2.8
ENV_Y_MIN = -10.05
ENV_Y_MAX = 10.05

def obq_line_distance(p3):
    p1 = np.array([3.0, 10.0])
    p2 = np.array([-0.4, -10.0])

    return np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1)

def in_env_collision(positions):
    if isinstance(positions, list):
        positions = np.array(positions)
    is_single_point = len(positions.shape) == 1

    if is_single_point:
        positions = positions[None, :]

    x = positions[:, 0]
    y = positions[:, 1]

    oblique_cond = 0.1 - obq_line_distance(positions)

    exceeds_env_bounds = np.array([False for _ in range(positions.shape[0])])

    exceeds_env_bounds[x < ENV_X_MIN] = True
    exceeds_env_bounds[y < ENV_Y_MIN] = True
    exceeds_env_bounds[y > ENV_Y_MAX] = True
    exceeds_env_bounds[oblique_cond <= 0] = True

    in_collision_with_stairs = np.all(
        [-2.8 <= x, x <= -0.06, -0.225 <= y, y <= 6.825], axis=0
    )

    in_collision = np.any([exceeds_env_bounds, in_collision_with_stairs], axis=0)

    if is_single_point:
        return in_collision[0]

    return in_collision

def get_random_pose():
    x = np.random.uniform(ENV_X_MIN, ENV_X_MAX)
    y = np.random.uniform(ENV_Y_MIN, ENV_Y_MAX)
    theta = np.random.uniform(-np.pi, np.pi)

    while in_env_collision(np.array([x, y])):
        x = np.random.uniform(ENV_X_MIN, ENV_X_MAX)
        y = np.random.uniform(ENV_Y_MIN, ENV_Y_MAX)

    return np.array([x, y, theta])


def convert_robot_orientation_to_quat(orientation):
    return utils.convert_euler_to_quat(np.array([0, 0, np.pi + orientation]))
