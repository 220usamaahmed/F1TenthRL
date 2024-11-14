from src.agent import Agent
from typing import Dict, Tuple


class DummyAgent(Agent):

    def take_action(self, obs: Dict) -> Tuple[float, float]:
        return (0.05, 1)
