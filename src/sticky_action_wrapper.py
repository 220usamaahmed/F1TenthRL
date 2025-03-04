import typing

import gymnasium
import numpy as np

from src.f110_sb_env import F110_SB_Env
from src.f110_sb_env_wrapper import F110_SB_Env_Wrapper


class StickyActionWrapper(F110_SB_Env_Wrapper):
    def __init__(
        self,
        env: F110_SB_Env_Wrapper,
        tick_rate=0.1,
        fine_rendering=False,
    ):
        print("Using StickyActionsWrapper")

        # if isinstance(env, F110_SB_Env):
        #     self._env = env
        # else:
        #     assert isinstance(env, F110_SB_Env_Wrapper)
        #     self._env = env.env

        self._env = env

        self._fine_rendering = fine_rendering
        self._ticks_per_step = int(tick_rate // self._env.TIMESTEP)

        self._current_action = None
        self._repeat_counter = 0

    @property
    def env(self):
        return self._env

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        if not self._fine_rendering:
            return self._step(action)
        else:
            return self._fine_step(action)

    def _step(self, action):
        obs = None
        total_reward = 0.0
        terminated = False
        truncated = False
        info = None

        t = 0
        for _ in range(self._ticks_per_step):
            t += 1
            obs, reward, terminated, truncated, info = self._env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        return obs, total_reward / t, terminated, truncated, info

    def _fine_step(self, action):
        if self._repeat_counter == 0:
            self._current_action = action
            self._repeat_counter = self._ticks_per_step - 1
        else:
            self._repeat_counter -= 1

        return self._env.step(self._current_action)
