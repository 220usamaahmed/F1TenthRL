import typing
import gym
import gymnasium
from gymnasium.spaces import Dict, Box
import numpy as np
from f110_gym.envs.base_classes import Integrator
from pyglet.gl import GL_LINES


class F110_SB_Env(gymnasium.Env):

    DEFAULT_NUM_BEAMS = 1080
    DEFAULT_FOV = 4.7

    def __init__(
        self,
        map: str,
        map_ext: str,
        lidar_params: typing.Dict,
        reset_pose: typing.Tuple[float, float],
    ):
        self.map = map
        self.map_ext = map_ext
        self.lidar_params = lidar_params
        self.reset_pose = reset_pose

        self._beam_rendering_enabled = False
        self._num_beams = lidar_params.get("num_beams", F110_SB_Env.DEFAULT_NUM_BEAMS)
        self._fov = lidar_params.get("fov", F110_SB_Env.DEFAULT_FOV)
        self._beam_gl_lines = []

        self.action_space = self._define_action_space()
        self.observation_space = self._define_observation_space()

        self._initialize_f110_gym()

    def _define_action_space(self):
        return Box(
            low=np.array([[-0.1, -0.1]]),
            high=np.array([[0.1, 0.1]]),
            shape=(1, 2),
            dtype=np.float32,
        )

    def _define_observation_space(self):
        # TODO: The high value of 30 should not be hard coded
        # TODO: Get low and high values of linear and angular velocities should not be hard coded
        # TODO: Should the pose x, y, theta also be included in observations?

        return Dict(
            {
                "scan": Box(low=0, high=30, shape=(self._num_beams,), dtype=np.float32),
                "linear_vel_x": Box(low=-10, high=10, shape=(), dtype=np.float32),
                "linear_vel_y": Box(low=-10, high=10, shape=(), dtype=np.float32),
                "angular_vel_z": Box(low=-10, high=10, shape=(), dtype=np.float32),
            }
        )

    def _initialize_f110_gym(self):
        self.env = gym.make(
            "f110_gym:f110-v0",
            map=self.map,
            map_ext=self.map_ext,
            num_agents=1,
            lidar_params=self.lidar_params,
            timestep=0.01,
            integrator=Integrator.RK4,
        )
        self.env.add_render_callback(self._render_callback)

    def _render_callback(self, env_renderer):
        if not len(self._beam_gl_lines):
            for _ in range(self.lidar_params.get("num_beams", 32)):
                gl_line = env_renderer.batch.add(
                    2,
                    GL_LINES,
                    None,
                    ("v2f/stream", (0, 0, 0, 0)),
                    ("c3B/stream", (255, 0, 0, 255, 0, 0)),
                )
                self._beam_gl_lines.append(gl_line)

    def _update_beam_gl_lines(self, obs):
        car_x = obs["poses_x"][0]
        car_y = obs["poses_y"][0]
        car_theta = obs["poses_theta"][0]
        scans = obs["scans"][0]

        for beam_i in range(self._num_beams):
            gl_line = self._beam_gl_lines[beam_i]
            theta = (
                car_theta + ((beam_i / self._num_beams) * self._fov) - (self._fov / 2)
            )

            end_x = car_x + np.cos(theta) * scans[beam_i]
            end_y = car_y + np.sin(theta) * scans[beam_i]

            # TODO: Find out why this is working with 50
            gl_line.vertices = [car_x * 50, car_y * 50, end_x * 50, end_y * 50]

    def _transform_obs_for_sb(self, obs):
        return {
            "scan": obs["scans"][0],
            "linear_vel_x": obs["linear_vels_x"][0],
            "linear_vel_y": obs["linear_vels_y"][0],
            "angular_vel_z": obs["ang_vels_z"][0],
        }

    def enable_beam_rendering(self):
        self._beam_rendering_enabled = True

    def disable_beam_rendering(self):
        self._beam_rendering_enabled = False

        for beam_i in range(self._num_beams):
            self._beam_gl_lines[beam_i].vertices = [0, 0, 0, 0]

    def reset(self, *, seed=None, options=None):
        obs, _, _, info = self.env.reset(np.array([self.reset_pose]))

        return self._transform_obs_for_sb(obs), info

    def step(self, action):
        print("action", action)
        obs, step_reward, done, info = self.env.step(np.array(action))

        if self._beam_rendering_enabled:
            self._update_beam_gl_lines(obs)

        # TODO: Pass truncated based on collison info
        return self._transform_obs_for_sb(obs), step_reward, done, done, info

    def render(self, mode: typing.Optional[str] = None):
        if mode is None:
            self.env.render()
        else:
            self.env.render(mode)
