import torch
import torch.nn as nn


class CNNExtractor(nn.Module):
    def __init__(self):
        super(CNNExtractor, self).__init__()

        self.features = nn.Sequential(

            # size is 96x96x3
            nn.Conv2d(3, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # size is 48x48x16
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # size is 24x24x32
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # size is 12x12x64
            nn.Conv2d(128, 64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # size is 6x6x128
            nn.Conv2d(64, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # size is 3x3x64
        )


        self.classifier = nn.Sequential(
            # 4 fully connected layers
            nn.Linear(32 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 49),

        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
