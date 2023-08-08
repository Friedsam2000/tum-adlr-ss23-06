import torch.nn as nn

class CNNExtractor(nn.Module):
    def __init__(self):
        super(CNNExtractor, self).__init__()

        # Assume input has shape (img_size, img_size, 12)
        # size 96x96x3
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(observation_space.shape[0], 32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
            # size 48x48x16

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            # size 24x24x32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            # size 12x12x64

        )

        self.objectGridClassifier = nn.Sequential(
            # 3 conv layer to 11x11x1
            # size 12x12x64
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            # size 12x12x32
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1),

            # size 12x12x1
            nn.Conv2d(16, 1, kernel_size=2, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),

            # size 11x11x1

        )

        self.positionClassifier = nn.Sequential(

            # 3 conv layer to 11x11x1
            # size 12x12x64
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            # size 6x6x32
            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            # size 3x3x16

            # Flatten
            nn.Flatten(),
            nn.Linear(3*3*16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, x):
        x = self.features(x)
        x_grid = self.objectGridClassifier(x)
        x_position = self.positionClassifier(x)
        return x_grid, x_position
