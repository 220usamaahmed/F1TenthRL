from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import gymnasium as gym


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        assert isinstance(observation_space, gym.spaces.Dict)
        super().__init__(observation_space, features_dim)

        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if key == "scan":
                # For the laser scan data
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                )
                total_concat_size += 64
            else:
                # For scalar values (linear_vel_x, linear_vel_y, angular_vel_z)
                extractors[key] = nn.Sequential(
                    nn.Linear(1, 8),  # Transform scalar to small feature vector
                    nn.ReLU(),
                )
                total_concat_size += 8

        self.extractors = nn.ModuleDict(extractors)

        # Add a final linear layer to get the desired feature dimension
        self.final_layer = nn.Sequential(
            nn.Linear(total_concat_size, features_dim), nn.ReLU()
        )

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # Process each observation key
        for key, extractor in self.extractors.items():
            if key in ["linear_vel_x", "linear_vel_y", "angular_vel_z"]:
                # Reshape scalar values to have a batch dimension
                observations[key] = observations[key].reshape(-1, 1)
            encoded_tensor_list.append(extractor(observations[key]))

        # Concatenate all processed tensors
        concatenated = th.cat(encoded_tensor_list, dim=1)
        return self.final_layer(concatenated)
