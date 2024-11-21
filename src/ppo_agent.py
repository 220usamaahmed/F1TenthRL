import typing
from src.agent import SBAgent, SBAgentLearningException
from src.f110_sb_env import F110_SB_Env
from src.feature_extractor import CustomCombinedExtractor
import numpy as np
from stable_baselines3 import PPO


class PPOAgent(SBAgent):

    def __init__(self, model: PPO):
        self._model = model

    @staticmethod
    def create(env: F110_SB_Env) -> SBAgent:
        policy_kwargs = {
            "features_extractor_class": CustomCombinedExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": dict(pi=[128, 64], vf=[128, 64]),
        }

        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=0.0003,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            tensorboard_log="./tensorboard_logs/",
        )

        return PPOAgent(model)

    @staticmethod
    def create_from_saved_model(model_path: str) -> SBAgent:
        model = PPO.load(model_path)
        return PPOAgent(model)

    def take_action(self, obs: typing.Dict, deterministic: bool = False) -> np.ndarray:
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return action

    def learn(self, total_timesteps=1000):
        try:
            self._model.learn(total_timesteps=total_timesteps)
        except Exception as e:
            raise SBAgentLearningException(e)

    def save_model(self, model_path: str):
        self._model.save(f"{model_path}.zip")
