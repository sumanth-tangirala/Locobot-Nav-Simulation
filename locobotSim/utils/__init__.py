import numpy as np
from scipy.spatial.transform import Rotation as R


def pos_distance(a, b):
    return np.linalg.norm(a - b)


def angular_diff(angles1, angles2):
    diff = angles1 - angles2

    is_scalar = False
    if not isinstance(diff, np.ndarray):
        is_scalar = True
        diff = np.array([diff])

    diff[np.abs(diff) > np.abs((2 * np.pi) + diff)] += 2 * np.pi

    return diff[0] if is_scalar else diff


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

    return r.as_euler("xyz", degrees=False)


def convert_scipy_quat_to_mujoco_quat(quat):
    return np.array([quat[3], quat[0], quat[1], quat[2]])


def convert_mujoco_quat_to_scipy_quat(quat):
    return np.array([quat[1], quat[2], quat[3], quat[0]])
