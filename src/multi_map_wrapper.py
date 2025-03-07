from random import choice
from typing import Callable, List, Tuple, Union

import gymnasium

from src.agent import Agent
from src.f110_sb_env import F110_SB_Env
from src.f110_sb_env_wrapper import F110_SB_Env_Wrapper
from src.utils import build_env, load_map_config


class MultiMapWrapper(F110_SB_Env_Wrapper):
    def __init__(
        self,
        env: Union[F110_SB_Env_Wrapper, F110_SB_Env],
        map_generator: Callable[
            [], Tuple[str, str, List[Tuple[float, float, float]], List["Agent"]]
        ],
    ):
        self._env = env

        self._map_generator = map_generator

    @property
    def env(self) -> F110_SB_Env:
        return self._env

    def reset(self, *, seed=None, options=None):
        if options is None:
            options = {}
        options["reset_map"] = True

        map, map_ext, reset_poses, other_agents = self._map_generator()
        options["map"] = map
        options["map_ext"] = map_ext
        options["reset_poses"] = reset_poses
        options["other_agents"] = other_agents

        # self.env.change_map(map, map_ext, reset_poses, other_agents)
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        return self._env.step(action)
