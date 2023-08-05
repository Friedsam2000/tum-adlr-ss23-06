import os
import torch
from torchvision import transforms
from cnnExtractor import CNNExtractor
from dataPreprocessor import load_data
import matplotlib.pyplot as plt
import random
import numpy as np

# Define transformation for the images, consistent with training script
transform = transforms.Compose([
    transforms.ToTensor()
])

# Close all existing figure windows
plt.close('all')

# Load the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(script_dir, '../img_data_generation/labels.csv')
images_dir_path = os.path.join(script_dir, '../img_data_generation')
dataset = load_data(csv_file=csv_file_path, images_dir=images_dir_path, transform=transform)

# Instantiate the model
model = CNNExtractor()

# Load the model weights
model.load_state_dict(torch.load('model.pth'))

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Define the loss function (consistent with training)
def custom_loss(predictions, grid_labels):
    predictions = predictions.view(predictions.size(0), -1)
    pos_weight = torch.full_like(grid_labels, 1)
    loss_grid = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(predictions, grid_labels)
    return loss_grid

# Variables to hold cumulative false positives, false negatives, and loss
cumulative_false_positives = 0
cumulative_false_negatives = 0
cumulative_loss = 0

# Number of predictions
num_predictions = 100

# Threshold for grid classification
threshold = 0.5

for _ in range(num_predictions):
    # Get a random index from the dataset
    random_index = random.randint(0, len(dataset) - 1)

    # Get the random sample from the dataset
    sample = dataset[random_index]
    image_tensor = sample['image'].to(device)

    # Pass the image through the model
    with torch.no_grad():
        prediction = model(image_tensor.unsqueeze(0)).squeeze()

    # Extract and print the agent's position, goal position, and neighboring grid
    predicted_neighboring_grid_logits = prediction.reshape(7, 7)
    predicted_neighboring_grid = torch.sigmoid(predicted_neighboring_grid_logits).tolist()

    # Extract the true agent's position, goal position, and neighboring grid
    true_neighboring_grid = torch.Tensor(sample['label'][4:]).reshape(7, 7).tolist()

    # Calculate false positives and false negatives
    false_positives = 0
    false_negatives = 0
    for i in range(7):
        for j in range(7):
            if predicted_neighboring_grid[i][j] >= threshold and true_neighboring_grid[i][j] < threshold:
                false_positives += 1
            if predicted_neighboring_grid[i][j] < threshold and true_neighboring_grid[i][j] >= threshold:
                false_negatives += 1

    # Add to cumulative counts
    cumulative_false_positives += false_positives
    cumulative_false_negatives += false_negatives

    # Convert true_neighboring_grid to a tensor
    true_neighboring_grid_tensor = torch.Tensor(true_neighboring_grid).to(device)

    # Calculate the loss
    loss = custom_loss(predicted_neighboring_grid_logits, true_neighboring_grid_tensor)

    # Add to cumulative loss
    cumulative_loss += loss.item()

# Print the predicted grid of the last iteration
print("Predicted NeighborGrid of last sample:")
predicted_grid_visual = [['O' if cell >= threshold else 'X' for cell in row] for row in predicted_neighboring_grid]
for row in predicted_grid_visual:
    print(" ".join(row))

# Calculate mean values
mean_false_positives = cumulative_false_positives / num_predictions
mean_false_negatives = cumulative_false_negatives / num_predictions
mean_loss = cumulative_loss / num_predictions

print("Mean number of false positives (falsely predicted obstacles):", mean_false_positives)
print("Mean number of false negatives (missed obstacles):", mean_false_negatives)
print("Mean loss of the predictions:", mean_loss)
