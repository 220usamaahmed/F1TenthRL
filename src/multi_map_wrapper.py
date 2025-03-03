from random import choice
from typing import Callable, List, Tuple

import gymnasium

from src.agent import Agent
from src.f110_sb_env import F110_SB_Env
from src.f110_sb_env_wrapper import F110_SB_Env_Wrapper
from src.utils import build_env, load_map_config


class MultiMapWrapper(F110_SB_Env_Wrapper):
    def __init__(
        self,
        env: F110_SB_Env_Wrapper,
        map_generator: Callable[
            [], Tuple[str, str, List[Tuple[float, float, float]], List["Agent"]]
        ],
    ):
        if isinstance(env, F110_SB_Env):
            self._env = env
        else:
            assert isinstance(env, F110_SB_Env_Wrapper)
            self._env = env.env

        self._map_generator = map_generator

    @property
    def env(self) -> F110_SB_Env:
        return self._env

    def reset(self, *, seed=None, options=None):
        map, map_ext, reset_poses, other_agents = self._map_generator()
        self.env.change_map(map, map_ext, reset_poses, other_agents)
        return self._env.reset(seed=seed, options=None)

    def step(self, action):
        return self._env.step(action)
