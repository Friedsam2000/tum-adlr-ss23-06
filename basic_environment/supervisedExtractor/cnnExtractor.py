import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataPreprocessor import load_data
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F  # Add this line

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        inter_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))  # Correctly using F.relu
        out = self.conv2(F.relu(self.bn2(out)))  # Correctly using F.relu
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

class CNNExtractor(nn.Module):
    def __init__(self, growth_rate=4, n_blocks=2, reduction=0.5):
        super(CNNExtractor, self).__init__()

        num_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)

        self.dense1 = self._make_dense(num_channels, growth_rate, n_blocks)
        num_channels += n_blocks * growth_rate
        out_channels = int(num_channels * reduction)
        self.trans1 = Transition(num_channels, out_channels)
        num_channels = out_channels

        self.dense2 = self._make_dense(num_channels, growth_rate, n_blocks)
        num_channels += n_blocks * growth_rate

        self.fc_obstacle = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 1, kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((7, 7))
        )

    def _make_dense(self, in_channels, growth_rate, n_blocks):
        layers = []
        for i in range(n_blocks):
            layers.append(Bottleneck(in_channels + i * growth_rate, growth_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        obstacle = self.fc_obstacle(x)
        obstacle = obstacle.view(obstacle.size(0), -1)
        return obstacle
