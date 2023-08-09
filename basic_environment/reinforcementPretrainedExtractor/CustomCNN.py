from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch


class CustomCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomCNNExtractor, self).__init__(observation_space, features_dim)

        self.cnn_grid = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 128, kernel_size=2, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01)
        )

        self.cnn_pos = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 128, kernel_size=2, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01)
        )

        self.linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 1 * 1 * 2, features_dim),
            nn.LeakyReLU(0.01)
        )

    def forward(self, observations):
        grid_data = observations[:, ::2, :, :]
        pos_data = observations[:, 1::2, :, :]

        grid_features = self.cnn_grid(grid_data)
        pos_features = self.cnn_pos(pos_data)

        grid_features = grid_features.view(grid_features.size(0), -1)
        pos_features = pos_features.view(pos_features.size(0), -1)
        combined_features = torch.cat([grid_features, pos_features], dim=1)

        return self.linear(combined_features)
