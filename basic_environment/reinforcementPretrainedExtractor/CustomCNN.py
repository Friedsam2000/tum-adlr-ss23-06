from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch

class CustomCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(CustomCNNExtractor, self).__init__(observation_space, features_dim)


        self.cnn_grid = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),   # [32, 11, 11]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                           # [32, 5, 5]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [64, 5, 5]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                           # [64, 2, 2]
            nn.Conv2d(64, 128, kernel_size=2, stride=1),            # [128, 1, 1]
            nn.ReLU()
        )

        self.cnn_pos = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),   # [32, 11, 11]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                           # [32, 5, 5]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [64, 5, 5]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                           # [64, 2, 2]
            nn.Conv2d(64, 128, kernel_size=2, stride=1),            # [128, 1, 1]
            nn.ReLU()
        )

        # Adjusted size calculation for 11x11 input. The final feature map size should be 1x1x128 for each cnn sequence.
        self.linear = nn.Sequential(
            nn.Linear(128*1*1*2, features_dim),                     # [features_dim]
            nn.ReLU(),
        )

    def forward(self, observations):
        grid_data = observations[:, ::2, :, :]
        pos_data = observations[:, 1::2, :, :]

        grid_features = self.cnn_grid(grid_data)                    # [batch_size, 128, 1, 1]
        pos_features = self.cnn_pos(pos_data)                       # [batch_size, 128, 1, 1]

        grid_features = grid_features.view(grid_features.size(0), -1)  # [batch_size, 128]
        pos_features = pos_features.view(pos_features.size(0), -1)     # [batch_size, 128]
        combined_features = torch.cat([grid_features, pos_features], dim=1)  # [batch_size, 256]

        return self.linear(combined_features)                              # [batch_size, features_dim]
