import os
import torch
from torch.utils.data import DataLoader
from cnnExtractor import CNNExtractor
from dataPreprocessor import load_data
from torch.utils.data import Subset
import torch.nn as nn
import numpy as np

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
    grid_loss_weight = 0
    pos_loss_weight = 1

    return grid_loss_weight * loss_grid + pos_loss_weight * loss_pos


# Load the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(script_dir, '../imageDataGeneration/labels.csv')
images_dir_path = os.path.join(script_dir, '../imageDataGeneration')
dataset = load_data(csv_file=csv_file_path, images_dir=images_dir_path)


# Limit the dataset to the first 10000 samples
max_images = 50
dataset = Subset(dataset, indices=range(max_images))

# Create a data loader with multiple workers
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)


# Load the model
model = CNNExtractor()
model_path = 'model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Custom function to print the grid
def print_grid(grid):
    for i in range(7):
        for j in range(7):
            print('X' if grid[i, j] == 1 else 'O', end=' ')
        print()
    print()

# Test the model
test_loss = 0.0
total_samples = 0
total_pos_error = 0.0
total_fp = 0
total_fn = 0
with torch.no_grad():
    for batch_idx, batch in enumerate(data_loader):
        images = batch['image'].to(device)
        grid_labels = batch['label'][:, 4:].clone().detach().float().to(device)  # Only obstacle grid, ignoring first 4 elements
        pos_labels = batch['label'][:, :4].clone().detach().float().to(device)  # Only the positions
        predictions_grid, predictions_pos = model(images)
        combined_loss = custom_loss(predictions_grid, predictions_pos, grid_labels, pos_labels)
        test_loss += combined_loss.item()
        total_samples += len(batch['image'])

        # Calculate the mean position error
        pos_error = torch.norm(predictions_pos - pos_labels, dim=1)
        total_pos_error += pos_error.sum().item()

        # Calculate the mean number of false positives and false negatives
        predicted_grid_binary = (torch.sigmoid(predictions_grid) > 0.5).float()  # Convert probabilities to binary values
        fp = (predicted_grid_binary == 1) & (grid_labels == 0)
        fn = (predicted_grid_binary == 0) & (grid_labels == 1)
        total_fp += fp.sum().item()
        total_fn += fn.sum().item()

        # Print the first sample grid and positions in the first batch
        if batch_idx == 0:
            true_grid = grid_labels[0].cpu().numpy().reshape(7, 7)
            predicted_grid = predicted_grid_binary[0].cpu().numpy().reshape(7, 7)
            true_positions = pos_labels[0].cpu().numpy()
            predicted_positions = predictions_pos[0].cpu().numpy()

            print("True grid:")
            print_grid(true_grid)
            print("Predicted grid:")
            print_grid(predicted_grid)
            print("True positions:", true_positions)
            print("Predicted positions:", np.round(predicted_positions))

print(f"Test loss: {test_loss / len(data_loader)}")
print(f"Test loss has been averaged over {total_samples} samples")
print(f"Mean position error: {total_pos_error / total_samples}")
print(f"Mean number of false positives per sample: {total_fp / total_samples}")
print(f"Mean number of false negatives per sample: {total_fn / total_samples}")