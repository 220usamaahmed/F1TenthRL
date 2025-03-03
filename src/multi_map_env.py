import typing

from random import choice
import gymnasium
from src.f110_sb_env import F110_SB_Env
from src.utils import load_map_config, build_env


class MultiMapEnv(gymnasium.Env):
    def __init__(
        self,
        map_names: typing.List[str],
        params: typing.Dict = {},
        lidar_params: typing.Dict = {},
        reward_params: typing.Dict = {},
    ):
        self._envs = {
            name: self._build_env(name, params, lidar_params, reward_params)
            for name in map_names
        }
        self._current_env = choice(self._envs)

    def _build_env(map_name, params, lidar_params, reward_params) -> F110_SB_Env:
        config = load_map_config(map_name)
        env = build_env(config, [], params, lidar_params, reward_params, False)

        return env

    def pick_env(self, map_name: str | None):
        assert map_name is None or map_name in self._envs.keys()

        map_name = map_name if map_name is not None else choice(self._envs.keys())
        self._current_env = self._envs[map_name]

    def reset(self, *, seed=None, options=None):
        self.pick_env()
        obs, info = self._current_env.reset()

    def step(self, action):
        obs, reward, terminated, truncated, info = self._current_env(action)

    def render(self, mode: typing.Optional[str] = None):
        self._current_env.render(mode)
