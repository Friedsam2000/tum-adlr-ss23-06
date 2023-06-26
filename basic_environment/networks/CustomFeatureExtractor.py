import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
import torch

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)

        # Assume input has shape (48, 48, 3)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(5, 16, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),

            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
