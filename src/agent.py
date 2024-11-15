from abc import ABC, abstractmethod
import typing


class Agent(ABC):

    @abstractmethod
    def take_action(self, obs: typing.Dict) -> typing.List[typing.List[float]]: ...
