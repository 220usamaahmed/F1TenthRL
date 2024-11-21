import typing
from abc import ABC, abstractmethod
import numpy as np

from src.f110_sb_env import F110_SB_Env


class Agent(ABC):

    @abstractmethod
    def take_action(
        self, obs: typing.Dict, deterministic: bool = True
    ) -> np.ndarray: ...


class SBAgent(Agent):

    @staticmethod
    @abstractmethod
    def create(env: F110_SB_Env) -> "SBAgent": ...

    @staticmethod
    @abstractmethod
    def create_from_saved_model(model_path: str) -> "SBAgent": ...

    @abstractmethod
    def learn(self, total_timesteps: int = 1000): ...

    @abstractmethod
    def save_model(self, model_path: str): ...


class SBAgentLearningException(Exception):

    def __init__(self, original_exception):
        self.original_exception = original_exception
        super().__init__(f"An error occurred: {str(original_exception)}")
