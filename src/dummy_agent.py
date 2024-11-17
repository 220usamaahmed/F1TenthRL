from src.agent import Agent
import typing
import numpy as np


class DummyAgent(Agent):

    def __init__(self, steer=1, speed=1):
        self.steer = steer
        self.speed = speed

    def take_action(self, obs: typing.Dict, deterministic=True) -> np.ndarray:
        return np.array([self.steer, self.speed])
