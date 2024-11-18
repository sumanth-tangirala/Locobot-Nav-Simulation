import mujoco as mj
from mujoco.glfw import glfw

import numpy as np


class Callbacks:
    def __init__(self, model, data, cam, scene, do_reset, apply_action):
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 1
        self.lasty = 0

        self.model = model
        self.data = data
        self.cam = cam
        self.scene = scene
        self.do_reset = do_reset
        self.apply_action = apply_action

        self.action = np.zeros(self.model.nu)

    def mouse_button(self, window, button, act, mods):
        self.button_left = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        )
        self.button_middle = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        )
        self.button_right = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        )

        glfw.get_cursor_pos(window)

    def mouse_move(self, window, xpos, ypos):
        dx = xpos - self.lastx
        dy = ypos - self.lasty

        self.lastx = xpos
        self.lasty = ypos

        if not (self.button_left or self.button_middle or self.button_right):
            return

        w, h = glfw.get_window_size(window)

        lshift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        rshift = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        shift = lshift or rshift

        if self.button_right:
            if shift:
                action = mj.mjtMouse.mjMOUSE_MOVE_H  # pylint: disable=no-member
            else:
                action = mj.mjtMouse.mjMOUSE_MOVE_V  # pylint: disable=no-member
        elif self.button_left:
            if shift:
                action = mj.mjtMouse.mjMOUSE_ROTATE_H  # pylint: disable=no-member
            else:
                action = mj.mjtMouse.mjMOUSE_ROTATE_V  # pylint: disable=no-member
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM  # pylint: disable=no-member

        mj.mjv_moveCamera(  # pylint: disable=no-member
            self.model,
            action,
            dx / h,  # pylint: disable=no-member
            dy / h,
            self.scene,
            self.cam,
        )

    def scroll(self, window, xoffset, yoffset):
        action = mj.mjtMouse.mjMOUSE_ZOOM  # pylint: disable=no-member
        mj.mjv_moveCamera(  # pylint: disable=no-member
            self.model,
            action,
            0.0,
            -0.05 * yoffset,  # pylint: disable=no-member
            self.scene,
            self.cam,
        )

    def handle_press_process(self, key, act):
        if key == glfw.KEY_W:
            if act == glfw.PRESS:
                self.action[0] += 0.1
        if key == glfw.KEY_S:
            if act == glfw.PRESS:
                self.action[0] -= 0.1
        if key == glfw.KEY_I:
            if act == glfw.PRESS:
                self.action[1] += 0.1
        if key == glfw.KEY_K:
            if act == glfw.PRESS:
                self.action[1] -= 0.1

        self.action = np.clip(self.action, -1, 1)

        self.apply_action(self.action)

    def keyboard(self, window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            mj.mj_resetData(self.model, self.data)  # pylint: disable=no-member
            mj.mj_forward(self.model, self.data)  # pylint: disable=no-member
            self.action = np.zeros(self.model.nu)
            self.do_reset()

        if act == glfw.PRESS and key == glfw.KEY_SPACE:
            self.action = np.zeros(self.model.nu)
            self.data.ctrl = self.action

        self.handle_press_process(key, act)
