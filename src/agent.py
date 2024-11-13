from abc import ABC, abstractmethod
from typing import Dict, Tuple


class Agent(ABC):

    @abstractmethod
    def take_action(self, obs: Dict) -> Tuple[float, float]: ...
