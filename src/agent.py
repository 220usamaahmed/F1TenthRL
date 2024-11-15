import typing
from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):

    @abstractmethod
    def take_action(self, obs: typing.Dict) -> np.ndarray: ...
