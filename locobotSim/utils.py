import numpy as np
from scipy.spatial.transform import Rotation as R


def pos_distance(a, b):
    return np.linalg.norm(a - b)


def convert_euler_to_quat(euler, is_scipy_quat=False):
    r = R.from_euler("xyz", euler, degrees=False)
    quat = r.as_quat()

    if not is_scipy_quat:
        quat = convert_scipy_quat_to_mujoco_quat(quat)

    return quat


def convert_quat_to_euler(quat, is_scipy_quat=False):
    if not is_scipy_quat:
        quat = convert_mujoco_quat_to_scipy_quat(quat)

    r = R.from_quat(quat)

    return r.as_euler("xyz", degrees=True)


def convert_scipy_quat_to_mujoco_quat(quat):
    return np.array([quat[3], quat[0], quat[1], quat[2]])


def convert_mujoco_quat_to_scipy_quat(quat):
    return np.array([quat[1], quat[2], quat[3], quat[0]])
