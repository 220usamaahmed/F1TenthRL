from src.agent import Agent
import typing
import numpy as np


class DummyAgent(Agent):

    def take_action(self, obs: typing.Dict) -> typing.List[typing.List[float]]:
        return np.array([[0.05, 1.0]])
