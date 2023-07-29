import torch
import torch.nn as nn

class CNNExtractor(nn.Module):
    def __init__(self):
        super(CNNExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Separate fully connected layers for position prediction
        self.fc_position = nn.Sequential(
            nn.Linear(32 * 22 * 22, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4), # Agent and goal position (x and y for both)
        )

        # Separate fully connected layers for obstacle classification
        self.fc_obstacle = nn.Sequential(
            nn.Linear(32 * 22 * 22, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 49), # Neighboring cells (7x7)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)

        # Separate forward passes for position and obstacle classification
        position = self.fc_position(x)
        obstacle = self.fc_obstacle(x)

        # Concatenate the outputs
        x = torch.cat((position, obstacle), dim=1)
        return x