import typing

from optuna.trial import Trial
from src.agent import SBAgent, SBAgentLearningException
from src.f110_sb_env import F110_SB_Env
from src.feature_extractor import CustomCombinedExtractor
import numpy as np
from stable_baselines3 import PPO


class PPOAgent(SBAgent):

    def __init__(self, model: PPO):
        self._model = model

    @staticmethod
    def create(
        env: F110_SB_Env,
        learning_rate=0.0003,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
        verbose=True,
        tensorboard_logs: typing.Union[str, None] = "./tensorboard_logs",
    ) -> SBAgent:
        policy_kwargs = {
            "features_extractor_class": CustomCombinedExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": net_arch,
        }

        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            tensorboard_log=tensorboard_logs,
        )

        return PPOAgent(model)

    @staticmethod
    def create_from_saved_model(model_path: str, env=None) -> SBAgent:
        model = PPO.load(model_path)
        if env is not None:
            model.set_env(env)
        return PPOAgent(model)

    def take_action(
        self, obs: typing.Dict, info: typing.Dict, deterministic: bool = False
    ) -> np.ndarray:
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return action

    def learn(self, total_timesteps=1000):
        self._model.learn(total_timesteps=total_timesteps)
        # try:
        #     self._model.learn(total_timesteps=total_timesteps)
        # except Exception as e:
        #     raise SBAgentLearningException(e)

    def save_model(self, model_path: str):
        self._model.save(f"{model_path}.zip")
        print(f"Model saved at: {model_path}")
