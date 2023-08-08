import torch.nn as nn

class CNNExtractor(nn.Module):
    def __init__(self):
        super(CNNExtractor, self).__init__()

        # Assume input has shape (img_size, img_size, 3)
        # size 96x96x3
        self.objectGridClassifier = nn.Sequential(
            # size 96x96x3
            nn.Conv2d(3, 8, kernel_size=4, stride=4),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            # size 24x24x32
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # size 12x12x16
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # size 6x6x32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # size 3x3x64
            nn.Conv2d(64, 11*11, kernel_size=2,stride=1),
            # size is 2x2x11*11
            nn.BatchNorm2d(11*11),


            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # size is 1x1x11*11



            nn.Flatten(),





        )


        self.positionClassifier = nn.Sequential(

            # size 96x96x3
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            # size 48x48x32
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

            # size 12x12x64
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # size 6x6x128
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # size 3x3x256

            # Flatten
            nn.Flatten(),
            nn.Linear(3*3*256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        x_grid = self.objectGridClassifier(x)
        x_position = self.positionClassifier(x)
        return x_grid, x_position
