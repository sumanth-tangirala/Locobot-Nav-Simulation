import numpy as np
import mujoco as mj
from mujoco.glfw import glfw

from locobotSim import utils
from locobotSim.env.locobot_env import LocobotEnv
from locobotSim.utils.callbacks import Callbacks

np.set_printoptions(suppress=True)


class LocobotVisualizer:
    def __init__(self):
        self.locobot = LocobotEnv(num_humans=5)
        self.locobot.reset()

        cam = mj.MjvCamera()
        opt = mj.MjvOption()

        glfw.init()
        window = glfw.create_window(800, 600, "Demo", None, None)
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        mj.mjv_defaultCamera(cam)
        mj.mjv_defaultOption(opt)
        scene = mj.MjvScene(self.locobot.model, maxgeom=10000)
        context = mj.MjrContext(
            self.locobot.model, mj.mjtFontScale.mjFONTSCALE_150.value
        )

        cb = Callbacks(
            self.locobot.model,
            self.locobot.data,
            cam,
            scene,
            self.locobot.reset,
            self.locobot.apply_action,
        )

        glfw.set_key_callback(window, cb.keyboard)
        glfw.set_cursor_pos_callback(window, cb.mouse_move)
        glfw.set_mouse_button_callback(window, cb.mouse_button)
        glfw.set_scroll_callback(window, cb.scroll)

        cam.azimuth = 180.0
        cam.elevation = -90.0
        cam.distance = 20.0
        cam.lookat = np.array([0.0, 0.0, 0.0])

        self.window = window
        self.cam = cam
        self.opt = opt
        self.scene = scene
        self.context = context

    def run(self):
        steps = 0
        angle = 0
        inc = 0.01
        while not glfw.window_should_close(self.window):
            steps += 1
            self.locobot.step(np.array([1.0, 1.0]))
            viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

            mj.mjv_updateScene(
                self.locobot.model,
                self.locobot.data,
                self.opt,
                None,
                self.cam,
                mj.mjtCatBit.mjCAT_ALL.value,
                self.scene,
            )
            mj.mjr_render(viewport, self.scene, self.context)

            glfw.swap_buffers(self.window)
            glfw.poll_events()


if __name__ == "__main__":
    visualizer = LocobotVisualizer()
    visualizer.run()
