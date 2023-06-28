import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)

        # Assume input has shape (16, 16, 3*num_frames_to_stack)
        self.cnn = nn.Sequential(

            nn.Conv2d(observation_space.shape[0], 8, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.1),

            nn.Conv2d(8, 8, kernel_size=2, stride=1, padding=1),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.1),

            nn.Conv2d(8, 64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            nn.Conv2d(64, 256, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),


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
