import typing
import numpy as np
from numpy import genfromtxt
from src.agent import Agent


class PlaybackAgent(Agent):

    def __init__(self, recording_path: str):
        self._recording = genfromtxt(recording_path, delimiter=",")
        self._current_timestep = 0

    def take_action(self, obs: typing.Dict, deterministic=True) -> np.ndarray:
        if self._current_timestep >= self._recording.shape[0]:
            return np.array([0, 0])
        else:
            action = self._recording[self._current_timestep, :]
            self._current_timestep += 1
            return action
