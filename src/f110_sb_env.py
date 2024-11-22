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
    DEFAULT_MAX_RANGE = 30.0
    DEFAULT_SV_MIN = -3.2
    DEFAULT_SV_MAX = 3.2
    DEFAULT_S_MIN = -0.4189
    DEFAULT_S_MAX = 0.4189
    DEFAULT_V_MIN = -5
    DEFAULT_V_MAX = 10
    ACTION_DAMPING_FACTORS = np.array([0.5, 0.5])

    def __init__(
        self,
        map: str,
        map_ext: str,
        reset_pose: typing.Tuple[float, float, float],
        params: typing.Dict = {},
        lidar_params: typing.Dict = {},
        record_actions=False,
    ):
        self.map = map
        self.map_ext = map_ext
        self.reset_pose = reset_pose
        self.record_actions = record_actions

        self._beam_rendering_enabled = False
        self._beam_gl_lines = []
        self._recorded_actions = [[]]
        self._previous_action = None

        self._num_beams = lidar_params.get("num_beams", F110_SB_Env.DEFAULT_NUM_BEAMS)
        self._fov = lidar_params.get("fov", F110_SB_Env.DEFAULT_FOV)
        self._max_range = lidar_params.get("max_range", F110_SB_Env.DEFAULT_MAX_RANGE)
        self._lidar_params = {
            "num_beams": self._num_beams,
            "fov": self._fov,
            "max_range": self._max_range,
        }

        self._sv_min = params.get("sv_min", F110_SB_Env.DEFAULT_SV_MIN)
        self._sv_max = params.get("sv_max", F110_SB_Env.DEFAULT_SV_MAX)
        self._s_min = params.get("s_min", F110_SB_Env.DEFAULT_S_MIN)
        self._s_max = params.get("s_max", F110_SB_Env.DEFAULT_S_MAX)
        self._v_min = params.get("v_min", F110_SB_Env.DEFAULT_V_MIN)
        self._v_max = params.get("v_max", F110_SB_Env.DEFAULT_V_MAX)
        self._params = {
            "sv_min": self._sv_min,
            "sv_max": self._sv_max,
            "s_min": self._s_min,
            "s_max": self._s_max,
            "v_min": self._v_min,
            "v_max": self._v_max,
        }

        assert self._s_min < 0 and self._s_max > 0
        assert self._v_min < 0 and self._v_max > 0
        self._s_normalization_factor = max(abs(self._s_min), self._s_max)
        self._v_normalization_factor = max(abs(self._v_min), self._v_max)
        self._action_scale_factors = np.array(
            [self._s_normalization_factor, self._v_normalization_factor]
        )

        self.action_space = self._define_action_space()
        self.observation_space = self._define_observation_space()

        self._initialize_f110_gym()

    def _define_action_space(self):
        # action: (steer, speed)
        s_min_normalized = self._s_min / self._s_normalization_factor
        s_max_normalized = self._s_max / self._s_normalization_factor

        v_min_normalized = self._v_min / self._v_normalization_factor
        v_max_normalized = self._v_max / self._v_normalization_factor

        return Box(
            low=np.array([s_min_normalized, v_min_normalized]),
            high=np.array([s_max_normalized, v_max_normalized]),
            dtype=np.float32,
        )

    def _define_observation_space(self):
        # TODO: Is the v_min and v_max really refering to linear velocities in x, y components?
        # TODO: Should the pose x, y, theta also be included in observations?

        return Dict(
            {
                "scan": Box(
                    low=0,
                    high=self._max_range,
                    shape=(self._num_beams,),
                    dtype=np.float32,
                ),
                "linear_vel_x": Box(
                    low=self._v_min, high=self._v_max, shape=(), dtype=np.float32
                ),
                "linear_vel_y": Box(
                    low=self._v_min, high=self._v_max, shape=(), dtype=np.float32
                ),
                "angular_vel_z": Box(
                    low=self._sv_min, high=self._sv_max, shape=(), dtype=np.float32
                ),
            }
        )

    def _initialize_f110_gym(self):
        self.env = gym.make(
            "f110_gym:f110-v0",
            map=self.map,
            map_ext=self.map_ext,
            num_agents=1,
            lidar_params=self._lidar_params,
            params=self._params,
            timestep=0.01,
            # integrator=Integrator.RK4,
            integrator=Integrator.Euler,
        )
        self.env.add_render_callback(self._render_callback)

    def _render_callback(self, env_renderer):
        # Adding LIDAR beam lines on first run
        if not len(self._beam_gl_lines):
            for _ in range(self._num_beams):
                gl_line = env_renderer.batch.add(
                    2,
                    GL_LINES,
                    None,
                    ("v2f/stream", (0, 0, 0, 0)),
                    ("c3B/stream", (255, 0, 0, 255, 0, 0)),
                )
                self._beam_gl_lines.append(gl_line)

        # Updating camera position
        car_vertices = env_renderer.cars[0].vertices
        x = np.mean(car_vertices[::2])
        y = np.mean(car_vertices[1::2])
        env_renderer.set_center(x, y)

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
        # TODO: Issue with inf in angular velocity
        a = obs["ang_vels_z"][0]
        a = min(a, self._sv_max)
        a = max(a, self._sv_min)

        return {
            "scan": obs["scans"][0],
            "linear_vel_x": obs["linear_vels_x"][0],
            "linear_vel_y": obs["linear_vels_y"][0],
            "angular_vel_z": a,
        }

    def _shape_reward(self, env_reward, obs):
        reward = env_reward

        if obs["collisions"][0] == 1.0:
            reward = -10
        else:
            velocity = obs["linear_vels_x"][0]
            reward = (1 / self._v_max) * velocity
            # angular_velocity = obs["ang_vels_z"][0]
            # reward += 0.1 * velocity - 0.1 * angular_velocity

        return reward

    def enable_beam_rendering(self):
        self._beam_rendering_enabled = True

    def disable_beam_rendering(self):
        self._beam_rendering_enabled = False

        for beam_i in range(self._num_beams):
            self._beam_gl_lines[beam_i].vertices = [0, 0, 0, 0]

    def enable_recording(self):
        self.record_actions = True
        self._recorded_actions = [[]]

    def disable_recording(self):
        self.record_actions = False
        self._recorded_actions = [[]]

    def reset(self, *, seed=None, options=None):
        obs, _, _, info = self.env.reset(np.array([self.reset_pose]))
        self._previous_action = None

        # TODO: Reset is called before training so there is an extra empty list
        if self.record_actions:
            self._recorded_actions.append([])

        return self._transform_obs_for_sb(obs), info

    def step(self, action):
        if self.record_actions:
            self._recorded_actions[-1].append(action)

        if self._previous_action is None:
            self._previous_action = action
        else:
            action = (
                self._previous_action * F110_SB_Env.ACTION_DAMPING_FACTORS
                + action * (1 - F110_SB_Env.ACTION_DAMPING_FACTORS)
            )

        scalled_actions = self._action_scale_factors * action

        obs, step_reward, done, info = self.env.step(scalled_actions.reshape(1, -1))

        if self._beam_rendering_enabled:
            self._update_beam_gl_lines(obs)

        transformed_obs = self._transform_obs_for_sb(obs)
        shaped_reward = self._shape_reward(step_reward, obs)
        return transformed_obs, shaped_reward, done, done, info

    def render(self, mode: typing.Optional[str] = None):
        if mode is None:
            self.env.render()
        else:
            self.env.render(mode)

    def get_recorded_actions(self):
        assert self.record_actions, "Env not configured to record actions"

        return self._recorded_actions
