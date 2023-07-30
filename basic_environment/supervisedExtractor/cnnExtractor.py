import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1),
            ))

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat((x, out), dim=1)
            # print(x.size())  # Print the shape after each layer
        return x


class CNNExtractor(nn.Module):
    def __init__(self):
        super(CNNExtractor, self).__init__()
        self.conv_layers = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size=3),

            DenseBlock(16, 4, 3), # Dense block with growth rate of 4 and 3 layers

            nn.Conv2d(28, 32, kernel_size=3), # 28 channels due to dense block concatenation
            nn.MaxPool2d(2),

            DenseBlock(32, 8, 3), # Another dense block
            nn.MaxPool2d(2),

        )

        spatial_dim = 23
        in_channels = 56

        self.fc_position = nn.Sequential(
            nn.Linear(in_channels * spatial_dim * spatial_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

        self.conv_obstacle = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Reduce spatial dimensions
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Further reduce spatial dimensions
            nn.Conv2d(128, 49, kernel_size=spatial_dim // 4),  # A convolution to reduce the spatial dimensions to 1x1
        )

    def forward(self, x):
        conv_out = self.conv_layers(x)
        print(conv_out.shape)  # Print shape after convolutional layers

        position_input = conv_out.view(conv_out.size(0), -1)
        position = self.fc_position(position_input)

        obstacle = self.conv_obstacle(conv_out)
        obstacle = obstacle.view(obstacle.size(0), -1)  # Flattening the output

        x = torch.cat((position, obstacle), dim=1)
        return x