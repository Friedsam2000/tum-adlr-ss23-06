import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(TransformerFeatureExtractor, self).__init__(observation_space, features_dim)

        self.encoder_layer = TransformerEncoderLayer(d_model=features_dim, nhead=4)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=2)

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.transformer_encoder(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.transformer_encoder(observations))
