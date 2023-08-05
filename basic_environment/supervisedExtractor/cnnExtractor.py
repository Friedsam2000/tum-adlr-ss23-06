import torch
import torch.nn as nn

class CNNExtractor(nn.Module):
    def __init__(self):
        super(CNNExtractor, self).__init__()

        self.features = nn.Sequential(
            # size is 96x96x3 (formula: (W−K+2P)/S+1)
            nn.Conv2d(3, 32, kernel_size=4, stride=4, padding=1),
            nn.ReLU(),

            # size is 24x24x32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # size is 24x24x64
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # size is 24x24x64
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # size is 12x12x64
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # size is 12x12x128
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # size is 12x12x32
            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # size is 12x12x8


        )

        self.classifier = nn.Sequential(
            # 4 fully connected layers
            nn.Linear(12 * 12*8, 256), # Here, you need to match the flattened size
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 49),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Reshape to [batch_size, 12 * 12 * 8]
        x = self.classifier(x)
        return x
