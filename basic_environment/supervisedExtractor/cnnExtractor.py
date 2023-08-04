import torch
import torch.nn as nn

class CNNExtractor(nn.Module):
    def __init__(self):
        super(CNNExtractor, self).__init__()

        self.features = nn.Sequential(
            # size is 96x96x3
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # size is 47x47x32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # size is 23x23x64
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # size is 11x11x128 formula: (W−K+2P)/S+1
            nn.Conv2d(128, 128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),

            # size is 10x10x128
            nn.Conv2d(128, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),

            # size is 9x9x64
            nn.Conv2d(64, 16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),

            # size is 8x8x32
            nn.Conv2d(16, 1, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),

            # size is 7x7x1


        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x
