import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from cnnExtractor import CNNExtractor
from dataPreprocessor import load_data
from torch.utils.data import Subset
from datetime import datetime
import torch.nn as nn


# Check for CUDA availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device} device')

# Define a custom loss function for grid prediction
def custom_loss(predictions_grid, predictions_pos, grid_labels, pos_labels):
    predictions_grid = predictions_grid.view(predictions_grid.size(0), -1)  # Reshaping predictions to match the target size
    pos_weight = torch.full_like(grid_labels, 1)
    loss_grid = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(predictions_grid, grid_labels)

    # MSE Loss for positions
    loss_pos = nn.MSELoss()(predictions_pos, pos_labels)

    # You can adjust the ratio of grid to position loss by using a different weight
    grid_loss_weight = 1.0
    pos_loss_weight = 0.001

    return grid_loss_weight * loss_grid + pos_loss_weight * loss_pos


# Define transformation for the images
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(script_dir, '../img_data_generation/labels.csv')
images_dir_path = os.path.join(script_dir, '../img_data_generation')
dataset = load_data(csv_file=csv_file_path, images_dir=images_dir_path, transform=transform)

# Create a data loader with multiple workers
data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)


# Load the model
model = CNNExtractor()
model_path = 'model.pth'
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# Test the model
test_loss = 0.0
with torch.no_grad():
    for batch in data_loader:
        images = batch['image'].to(device)
        grid_labels = batch['label'][:, 4:].clone().detach().float().to(device)  # Only obstacle grid, ignoring first 4 elements
        pos_labels = batch['label'][:, :4].clone().detach().float().to(device)  # Only the positions
        predictions_grid, predictions_pos = model(images)
        combined_loss = custom_loss(predictions_grid, predictions_pos, grid_labels, pos_labels)
        test_loss += combined_loss.item()

print(f"Test loss: {test_loss / len(data_loader)}")
