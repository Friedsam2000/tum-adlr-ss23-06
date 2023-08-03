import torch
import torch.nn as nn

class CNNExtractor(nn.Module):
    def __init__(self):
        super(CNNExtractor, self).__init__()

        self.fc_obstacle = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size=3, padding=1), # Output size: 96x96x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.5), # Added Dropout here

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Output size: 48x48x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.5), # Added Dropout here

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Output size: 48x48x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5), # Added Dropout here

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Output size: 24x24x128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5), # Added Dropout here

            nn.Conv2d(128, 256, kernel_size=3, padding=1), # Output size: 24x24x256
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5), # Added Dropout here

            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1), # Output size: 12x12x128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5), # Added Dropout here

            nn.Conv2d(128, 1, kernel_size=2),  # Output size: 11x11x1
            nn.AdaptiveAvgPool2d((7, 7)) # Output size: 7x7x1
        )

    def forward(self, x):
        obstacle = self.fc_obstacle(x)
        obstacle = obstacle.view(obstacle.size(0), -1) # Reshape to flatten the grid
        return obstacle
