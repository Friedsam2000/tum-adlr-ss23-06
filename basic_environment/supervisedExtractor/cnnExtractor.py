import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataPreprocessor import load_data
from torch.utils.tensorboard import SummaryWriter

class CNNExtractor(nn.Module):
    def __init__(self):
        super(CNNExtractor, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc_obstacle = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(128, 1, kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((7, 7))
        )

    def forward(self, x):
        x = self.conv_layers(x)
        obstacle = self.fc_obstacle(x)
        obstacle = obstacle.view(obstacle.size(0), -1)
        return obstacle

