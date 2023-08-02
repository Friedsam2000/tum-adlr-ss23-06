import torch
import torch.nn as nn

class CNNExtractor(nn.Module):
    def __init__(self):
        super(CNNExtractor, self).__init__()

        self.fc_obstacle = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output size: 48x48x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output size: 24x24x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3),  # Output size: 8x8x64
            nn.Conv2d(64, 1, kernel_size=2),  # Output size: 7x7x1
            nn.Sigmoid()  # To squash values between 0 and 1
        )

    def forward(self, x):
        obstacle = self.fc_obstacle(x)
        obstacle = obstacle.view(obstacle.size(0), -1) # Reshape to flatten the grid
        return obstacle
