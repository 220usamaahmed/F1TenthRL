import typing
import gym
import gymnasium
from gymnasium.spaces import Dict, Box
import numpy as np
from f110_gym.envs.base_classes import Integrator
from pyglet.gl import GL_LINES
from collections import deque

if typing.TYPE_CHECKING:
    from src.agent import Agent


class F110_SB_Env(gymnasium.Env):
    TIMESTEP = 0.01
    # DEFAULT_NUM_BEAMS = 1081
    # DEFAULT_MAX_RANGE = 30.0
    DEFAULT_NUM_BEAMS = 135
    DEFAULT_MAX_RANGE = 5.0
    DEFAULT_FOV = 2.3499999046325684 * 2
    DEFAULT_SV_MIN = -3.2
    DEFAULT_SV_MAX = 3.2
    DEFAULT_S_MIN = -0.4189
    DEFAULT_S_MAX = 0.4189
    DEFAULT_V_MIN = 1.0
    DEFAULT_V_MAX = 5.0
    DEFAULT_WIDTH = 0.3
    DEFAULT_LENGTH = 0.51
    ACTION_DAMPING_FACTORS = np.array([1.0, 1.0])
    EGO_IDX = 0
    MAX_EPOCHS = 6000
    MAX_STILL_STEPS = 100
    STILL_THRESHOLD = 0.1

    def __init__(
        self,
        map: str,
        map_ext: str,
        reset_poses: typing.List[typing.Tuple[float, float, float]],
        other_agents: typing.List["Agent"],
        params: typing.Dict = {},
        lidar_params: typing.Dict = {},
        reward_params: typing.Dict = {},
        record=False,
    ):
        assert (
            len(reset_poses) == len(other_agents) + 1
        ), f"{len(reset_poses)} reset pose(s) given but there are {len(other_agents) + 1} agent(s)"

        self._map = map
        self._map_ext = map_ext
        self._reset_poses = reset_poses
        self._other_agents = other_agents
        self.record = record

        self._num_agents = 1 + len(other_agents)

        self._beam_rendering_enabled = False
        self._beam_gl_lines = []
        self._direction_gl_line = None

        self._recorded_actions = []
        self._recorded_rewards = []
        self._recorded_observations = []
        self._recorded_info = []

        self._previous_obs = None
        self._previous_info = None
        self._previous_actions = None

        self._previous_poses = deque(maxlen=self.MAX_STILL_STEPS)
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
        self.width = params.get("width", F110_SB_Env.DEFAULT_WIDTH)
        self.length = params.get("length", F110_SB_Env.DEFAULT_LENGTH)
        self._params = {
            "sv_min": self._sv_min,
            "sv_max": self._sv_max,
            "s_min": self._s_min,
            "s_max": self._s_max,
            "v_min": self._v_min,
            "v_max": self._v_max,
            "width": self.width,
            "length": self.length,
        }

        self._reward_parms = reward_params

        # self._s_normalization_factor = max(abs(self._s_min), self._s_max)
        # self._v_normalization_factor = max(abs(self._v_min), self._v_max)
        # self._action_scale_factors = np.array(
        #     [self._s_normalization_factor, self._v_normalization_factor]
        # )

        self.action_space = self._define_action_space()
        self.observation_space = self._define_observation_space()

        self._initialize_f110_gym()
        self._epochs = 0

    def change_map(
        self,
        map: str,
        map_ext: str,
        reset_poses: typing.List[typing.Tuple[float, float, float]],
        other_agents: typing.List["Agent"],
    ):
        self._map = map
        self._map_ext = map_ext
        self._reset_poses = reset_poses
        self._other_agents = other_agents

        self._initialize_f110_gym()

    def _define_action_space(self):
        # action: (steer, speed)
        # s_min_normalized = self._s_min / self._s_normalization_factor
        # s_max_normalized = self._s_max / self._s_normalization_factor

        # v_min_normalized = self._v_min / self._v_normalization_factor
        # v_max_normalized = self._v_max / self._v_normalization_factor

        return Box(
            low=np.array([0, 0]),
            high=np.array([+1, +1]),
            dtype=np.float32,
        )

    def _define_observation_space(self):
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
                # "linear_vel_y": Box(
                #     low=self._v_min, high=self._v_max, shape=(), dtype=np.float32
                # ),
                # "angular_vel_z": Box(
                #     low=self._sv_min, high=self._sv_max, shape=(), dtype=np.float32
                # ),
            }
        )

    def _initialize_f110_gym(self):
        self._env = gym.make(
            "f110_gym:f110-v0",
            map=self._map,
            map_ext=self._map_ext,
            num_agents=self._num_agents,
            ego_idx=self.EGO_IDX,
            lidar_params=self._lidar_params,
            params=self._params,
            timestep=self.TIMESTEP,
            # integrator=Integrator.RK4,
            integrator=Integrator.Euler,
        )
        self._env.add_render_callback(self._render_callback)

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

        # Add direction line
        if not self._direction_gl_line:
            self._direction_gl_line = env_renderer.batch.add(
                2,
                GL_LINES,
                None,
                ("v2f/stream", (0, 0, 0, 0)),
                ("c3B/stream", (255, 255, 255, 0, 255, 0)),
            )

        # Updating camera position
        car_vertices = env_renderer.cars[self.EGO_IDX].vertices
        x = np.mean(car_vertices[::2])
        y = np.mean(car_vertices[1::2])
        env_renderer.set_center(x, y)

    def _update_beam_gl_lines(self, obs):
        car_x = obs["poses_x"][self.EGO_IDX]
        car_y = obs["poses_y"][self.EGO_IDX]
        car_theta = obs["poses_theta"][self.EGO_IDX]
        scans = obs["scans"][self.EGO_IDX]

        for beam_i in range(self._num_beams):
            gl_line = self._beam_gl_lines[beam_i]
            theta = (
                car_theta + ((beam_i / self._num_beams) * self._fov) - (self._fov / 2)
            )

            end_x = car_x + np.cos(theta) * scans[beam_i]
            end_y = car_y + np.sin(theta) * scans[beam_i]

            # TODO: Find out why this is working with 50
            gl_line.vertices = [car_x * 50, car_y * 50, end_x * 50, end_y * 50]

        pos_x_0 = self._previous_obs["poses_x"][self.EGO_IDX]
        pos_y_0 = self._previous_obs["poses_y"][self.EGO_IDX]
        pos_x_1 = obs["poses_x"][self.EGO_IDX]
        pos_y_1 = obs["poses_y"][self.EGO_IDX]

        theta = np.arctan2(pos_y_1 - pos_y_0, pos_x_1 - pos_x_0)
        mag = (pos_y_0 - pos_y_1) ** 2 + (pos_x_0 - pos_x_1) ** 2

        dx = 50 * np.cos(theta)
        dy = 50 * np.sin(theta)

        self._direction_gl_line.vertices = [
            car_x * 50,
            car_y * 50,
            (car_x * 50) + dx,
            (car_y * 50) + dy,
        ]

    def _transform_obs_and_info_for_sb(
        self, obs, info, idx=EGO_IDX
    ) -> typing.Tuple[dict, dict]:
        # TODO: Issue with inf in angular velocity
        a = obs["ang_vels_z"][idx]
        a = min(a, self._sv_max)
        a = max(a, self._sv_min)

        transformed_obs = {
            "scan": obs["scans"][idx],
            "linear_vel_x": obs["linear_vels_x"][idx],
            # "scan": obs["scans"][idx] / self._max_range,
            # "linear_vel_x": obs["linear_vels_x"][idx] / self._v_max,
            # "linear_vel_y": obs["linear_vels_y"][idx],
            # "angular_vel_z": a,
        }

        # assert not np.isnan(transformed_obs["scan"]).any(), "NaN in scans"
        # assert not np.isnan(transformed_obs["linear_vel_x"]), "NaN in lin vel x"
        # assert not np.isnan(transformed_obs["linear_vel_y"]), "NaN in lin vel y"

        transformed_info = {
            "pose_x": obs["poses_x"][idx],
            "pose_y": obs["poses_y"][idx],
            "pose_theta": obs["poses_theta"][idx],
            "lap_time": obs["lap_times"][idx],
            "lap_count": obs["lap_counts"][idx],
            "checkpoint_done": info["checkpoint_done"][idx],
        }

        return transformed_obs, transformed_info

    def _reset_recording(self):
        self._recorded_actions = []
        self._recorded_rewards = []
        self._recorded_observations = []
        self._recorded_info = []

    def _shape_reward(self, action, env_reward, obs, info, idx=EGO_IDX) -> float:
        if info["checkpoint_done"][idx]:
            reward = +100
        elif obs["collisions"][idx] == 1.0:
            reward = -100
        else:
            # reward = 0
            distance_to_boundary = np.min(obs["scans"][idx])
            r_dist = -1 if distance_to_boundary < 0.5 else 1

            velocity = obs["linear_vels_x"][idx]
            vel_norm = velocity / self._v_max
            r_vel = vel_norm

            reward = r_vel + r_dist

        return reward

        # return self._r7(action, obs, info, idx)

    def _r7(self, action, obs, info, idx):
        # TODO: Use the reward params

        """
        Generic reward function that weights each component based on reward
        paramters passed to the environment.

        The following components are considered:
            - x linear velocity
            - y linear velocity
            - z angular velocity
            - change in z angular velocity
            - min distance to boundry
            - change in min distance to boundry
            - others?
        """
        if info["checkpoint_done"][idx]:
            reward = +1000
        elif obs["collisions"][idx] == 1.0:
            reward = -1000
        elif self._check_truncated():
            reward = -50
        else:
            distance_to_boundary = np.min(obs["scans"][idx])
            velocity = obs["linear_vels_x"][idx]
            angular_velocity = obs["ang_vels_z"][idx]
            previous_angular_velocity = (
                0
                if self._previous_obs is None
                else self._previous_obs["ang_vels_z"][idx]
            )
            angular_velocity_delta = angular_velocity - previous_angular_velocity

            vel_norm = velocity / self._v_max
            dist_norm = distance_to_boundary / self._max_range
            ang_vel_norm = abs(angular_velocity) / self._sv_max
            ang_vel_del_norm = abs(angular_velocity_delta) / (2 * self._sv_max)
            steer_norm = action[0]

            r_vel = vel_norm
            r_dist = -5 if distance_to_boundary < 0.5 else 0
            r_a_vel = ang_vel_norm
            r_a_vel_delta = 1 - min(1, ang_vel_del_norm)
            r_steer = 1 - abs(steer_norm)

            # reward = 0.2 * r_vel + 0.8 * r_dist + 0.0 * r_a_vel + 0.0 * r_a_vel_delta

            # r_vel = 1 if vel_norm > 0.1 else 0

            reward = r_vel + 0.0 * r_steer + r_dist

        return reward

    def enable_beam_rendering(self):
        self._beam_rendering_enabled = True

    def disable_beam_rendering(self):
        self._beam_rendering_enabled = False

        for beam_i in range(self._num_beams):
            self._beam_gl_lines[beam_i].vertices = [0, 0, 0, 0]

    def enable_recording(self):
        self.record = True
        self._reset_recording()

    def disable_recording(self):
        self.record = False
        self._reset_recording()

    def reset(self, *, seed=None, options=None):
        if options is not None and options.get("reset_map", False):
            map = options.get("map")
            map_ext = options.get("map_ext")
            reset_poses = options.get("reset_poses")
            other_agents = options.get("other_agents")

            self.change_map(map, map_ext, reset_poses, other_agents)

        self._epochs = 0
        self._previous_poses = deque(maxlen=self.MAX_STILL_STEPS)

        obs, reward, _, info = self._env.reset(np.array(self._reset_poses))
        self._previous_obs = obs
        self._previous_info = info
        self._previous_actions = None

        self._previous_poses.append(
            (
                obs["poses_x"][self.EGO_IDX],
                obs["poses_y"][self.EGO_IDX],
            )
        )

        transformed_obs, transformed_info = self._transform_obs_and_info_for_sb(
            obs, info
        )
        shaped_reward = self._shape_reward([0.0, 0.0], reward, obs, info)

        if self.record:
            self._reset_recording()
            self._recorded_rewards.append(shaped_reward)
            self._recorded_observations.append(transformed_obs)
            self._recorded_info.append(transformed_info)

        return transformed_obs, transformed_info

    def _get_actions(self, ego_action):
        def _map_value(value, current_range, desired_range):
            """Maps a value from its current range to a desired range."""
            cur_min, cur_max = current_range
            des_min, des_max = desired_range
            return des_min + (value - cur_min) * (des_max - des_min) / (
                cur_max - cur_min
            )

        # TODO: Ignoring all other agents for now
        # TODO: Ignoring dampning for now
        # TODO: Hard coded Box highs and lows used

        steering_angle = ego_action[0]
        speed = ego_action[1]

        scaled_steering_angle = _map_value(
            steering_angle, (0, +1), (self._s_min, self._s_max)
        )
        scaled_speed = _map_value(speed, (0, +1), (self._v_min, self._v_max))

        # print("model_output", ego_action)
        # print("scaled_action", [scaled_steering_angle, scaled_speed])
        # print()

        all_actions = np.array([[scaled_steering_angle, scaled_speed]])
        self._previous_actions = all_actions

        return all_actions

        # all_actions = [ego_action]
        # for i, agent in enumerate(self._other_agents):
        #     current_transformed_obs, current_transformed_info = (
        #         self._transform_obs_and_info_for_sb(
        #             self._previous_obs, self._previous_info, i
        #         )
        #     )
        #     all_actions.append(
        #         agent.take_action(current_transformed_obs, current_transformed_info)
        #     )
        # all_actions = np.array(all_actions)

        # if self._previous_actions is None:
        #     damped_actions = all_actions
        # else:
        #     damped_actions = (
        #         self._previous_actions * (1 - F110_SB_Env.ACTION_DAMPING_FACTORS)
        #         + all_actions * F110_SB_Env.ACTION_DAMPING_FACTORS
        #     )
        # self._previous_actions = damped_actions

        # scalled_actions = self._action_scale_factors * damped_actions

        # return scalled_actions

    def _check_if_still(self):
        if len(self._previous_poses) < self.MAX_STILL_STEPS:
            return False

        pose_a = self._previous_poses[0]
        pose_b = self._previous_poses[-1]

        distance = (pose_a[0] - pose_b[0]) ** 2 + (pose_a[1] - pose_b[1]) ** 2

        return distance < self.STILL_THRESHOLD

    def _check_truncated(self):
        return self._epochs > F110_SB_Env.MAX_EPOCHS or self._check_if_still()

    def step(self, action):
        self._epochs += 1

        actions = self._get_actions(action)

        obs, step_reward, done, info = self._env.step(actions)
        truncated = self._check_truncated()

        transformed_obs, transformed_info = self._transform_obs_and_info_for_sb(
            obs, info
        )
        shaped_reward = self._shape_reward(action, step_reward, obs, info)
        terminated = done or obs["collisions"][self.EGO_IDX] == 1.0

        if self.record:
            self._recorded_actions.append(action)
            self._recorded_rewards.append(shaped_reward)
            self._recorded_observations.append(transformed_obs)
            self._recorded_info.append(transformed_info)

        if self._beam_rendering_enabled:
            self._update_beam_gl_lines(obs)

        self._previous_obs = obs
        self._previous_info = info

        self._previous_poses.append(
            (
                obs["poses_x"][self.EGO_IDX],
                obs["poses_y"][self.EGO_IDX],
            )
        )

        return (
            transformed_obs,
            shaped_reward,
            terminated,
            truncated,
            transformed_info,
        )

    def render(self, mode: typing.Optional[str] = None):
        if mode is None:
            self._env.render()
        else:
            self._env.render(mode)

    def get_recording(self):
        assert self.record, "Env not configured to record actions"

        return (
            self._recorded_actions,
            self._recorded_rewards,
            self._recorded_observations,
            self._recorded_info,
        )
