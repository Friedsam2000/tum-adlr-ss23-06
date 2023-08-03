import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNExtractor(nn.Module):
    def __init__(self):
        super(CNNExtractor, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # Output size: 96x96x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output size: 48x48x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output size: 48x48x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output size: 24x24x128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.agent_location = nn.Conv2d(128, 1, kernel_size=1)

        self.fc_obstacle = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Output size: 24x24x256
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),  # Output size: 12x12x128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(128, 1, kernel_size=2),  # Output size: 11x11x1
            nn.AdaptiveAvgPool2d((7, 7))  # Output size: 7x7x1
        )

    def forward(self, x):
        x = self.conv_layers(x)
        agent_location = self.agent_location(x)
        agent_location = F.softmax(agent_location.view(agent_location.size(0), -1), dim=1).view(agent_location.size(0),
                                                                                                1, x.size(2), x.size(3))

        # Create the coordinates tensor with the same batch size as the input
        agent_coords = torch.stack(
            torch.meshgrid(torch.arange(agent_location.size(2)), torch.arange(agent_location.size(3))), dim=0).float()
        agent_coords = agent_coords.to(agent_location.device).permute(1, 2, 0).unsqueeze(0)

        # Multiply agent_location with agent_coords
        agent_coords = (agent_location.unsqueeze(-1) * agent_coords).sum(dim=(2, 3))

        cropped_window = self.crop_window(x, agent_coords)
        obstacle = self.fc_obstacle(cropped_window)
        obstacle = obstacle.view(obstacle.size(0), -1)  # Reshape to flatten the grid
        return obstacle

    def crop_window(self, x, coords):
        batch_size, _, height, width = x.size()
        scale = torch.tensor([width // 24, height // 24], device=x.device).view(1, 2)
        theta = torch.zeros(batch_size, 2, 3, device=x.device)
        theta[:, 0, 0] = scale[0, 0]
        theta[:, 1, 1] = scale[0, 1]
        theta[:, :, 2] = ((coords.float() - scale / 2) * 2 / scale - 1).squeeze(1)
        grid = F.affine_grid(theta, size=(batch_size, x.size(1), 24, 24))
        cropped_window = F.grid_sample(x, grid)
        return cropped_window
