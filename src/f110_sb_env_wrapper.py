import typing
from abc import ABC, abstractmethod

import gymnasium

from src.f110_sb_env import F110_SB_Env


class F110_SB_Env_Wrapper(ABC, gymnasium.Env):
    @property
    @abstractmethod
    def env(self) -> F110_SB_Env:
        pass

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def record(self):
        return self.env.record

    def render(self, mode: typing.Optional[str] = None):
        self.env.render(mode=mode)

    def enable_beam_rendering(self):
        self.env.enable_beam_rendering()

    def disable_beam_rendering(self):
        self.env.disable_beam_rendering()

    def enable_recording(self):
        self.env.enable_recording()

    def disable_recording(self):
        self.env.disable_recording()

    def get_recording(self):
        return self.env.get_recording()

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self, seed=None, options=None):
        pass
