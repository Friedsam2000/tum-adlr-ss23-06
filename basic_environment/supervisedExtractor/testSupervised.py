import os
import torch
from torchvision import transforms
from cnnExtractor import CNNExtractor
from dataPreprocessor import load_data
import matplotlib.pyplot as plt
import random

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

# Get a random index from the dataset
random_index = random.randint(0, len(dataset) - 1)

# Instantiate the model
model = CNNExtractor()

# Load the model weights
model.load_state_dict(torch.load('model.pth'))

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

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

# Show the image
image_numpy = image_tensor.cpu().permute(1, 2, 0).numpy() # Convert to CPU for visualization
plt.imshow(image_numpy)
plt.axis('off') # To turn off axes
plt.show()

# Visualize the predicted and true grids
threshold = 0.5
print("Predicted NeighborGrid:")
predicted_grid_visual = [['O' if cell >= threshold else 'X' for cell in row] for row in predicted_neighboring_grid]
for row in predicted_grid_visual:
    print(" ".join(row))
print("True NeighborGrid:")
true_grid_visual = [['O' if cell >= threshold else 'X' for cell in row] for row in true_neighboring_grid]
for row in true_grid_visual:
    print(" ".join(row))

# Calculate false positives and false negatives
false_positives = 0
false_negatives = 0
for i in range(7):
    for j in range(7):
        if predicted_neighboring_grid[i][j] >= threshold and true_neighboring_grid[i][j] < threshold:
            false_positives += 1
        if predicted_neighboring_grid[i][j] < threshold and true_neighboring_grid[i][j] >= threshold:
            false_negatives += 1

print("Number of false positives (falsely predicted obstacles):", false_positives)
print("Number of false negatives (missed obstacles):", false_negatives)


# Define the loss function (consistent with training)
def custom_loss(predictions, grid_labels):
    predictions = predictions.view(predictions.size(0), -1)
    pos_weight = torch.full_like(grid_labels, 1)
    loss_grid = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(predictions, grid_labels)
    return loss_grid

# Convert true_neighboring_grid to a tensor
true_neighboring_grid_tensor = torch.Tensor(true_neighboring_grid).to(device)

# Calculate the loss
loss = custom_loss(predicted_neighboring_grid_logits, true_neighboring_grid_tensor)

# Print the loss
print("Loss of the prediction:", loss.item())
