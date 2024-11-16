import typing
from src.agent import Agent
from src.feature_extractor import CustomCombinedExtractor
import numpy as np
from stable_baselines3 import PPO


class PPOAgent(Agent):

    def __init__(self, env):
        self._model = self._create_model(env)

    def _create_model(self, env):
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

        return model

    def take_action(self, obs: typing.Dict, deterministic: bool = False) -> np.ndarray:
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return action

    def learn(self):
        self._model.learn(total_timesteps=200000)
