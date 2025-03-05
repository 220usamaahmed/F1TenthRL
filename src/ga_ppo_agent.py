import typing
import numpy as np
from src.ppo_agent import PPOAgent
from stable_baselines3 import PPO
import torch
from copy import deepcopy


class GA_PPO_Agent:

    MUTATION_STD_DEV = 0.002

    def __init__(self, proto_agent: PPOAgent):
        self._agent = self._mutate(deepcopy(proto_agent.get_model()))

    def _mutate(self, proto_agent: PPO) -> PPOAgent:
        # actor = proto_agent.mlp_extractor.policy_net
        action_net = proto_agent.policy.action_net

        with torch.no_grad():
            action_net.weight += (
                torch.randn_like(action_net.weight) * self.MUTATION_STD_DEV
            )
            action_net.bias += torch.randn_like(action_net.bias) * self.MUTATION_STD_DEV

        return PPOAgent(proto_agent)

    def take_action(
        self, obs: typing.Dict, info: typing.Dict, deterministic: bool = False
    ) -> np.ndarray:
        return self._agent.take_action(obs, info, deterministic=deterministic)

    def get_mutated_agent(self) -> PPOAgent:
        return self._agent
