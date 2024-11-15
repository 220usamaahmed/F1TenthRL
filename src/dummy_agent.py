from src.agent import Agent
import typing
import numpy as np


class DummyAgent(Agent):

    def take_action(self, obs: typing.Dict) -> np.ndarray:
        return np.array([0.05, 1.0])
