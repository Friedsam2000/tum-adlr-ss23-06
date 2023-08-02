import os
import torch
from torchvision import transforms
from cnnExtractor import CNNExtractor
from dataPreprocessor import load_data
import matplotlib.pyplot as plt
import random

# Define transformation for the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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
model.eval()

# Get the random sample from the dataset
sample = dataset[random_index]
image_tensor = sample['image']
image_numpy = image_tensor.permute(1, 2, 0).numpy() # Converting to a format suitable for displaying
image_numpy = (image_numpy + 1) / 2.0 # Denormalizing the image

# Pass the image through the model
model.eval() # Set the model to evaluation mode
with torch.no_grad():
    prediction = model(image_tensor.unsqueeze(0)).squeeze()

# Extract and print the agent's position, goal position, and neighboring grid
predicted_neighboring_grid_logits = prediction.reshape(7, 7)
predicted_neighboring_grid = torch.sigmoid(predicted_neighboring_grid_logits).tolist()

# Extract the true agent's position, goal position, and neighboring grid
true_neighboring_grid = sample['label'][4:].reshape(7, 7).tolist()

# Show the image
plt.imshow(image_numpy)
plt.axis('off') # To turn off axes
plt.show()

print("Predicted NeighborGrid:")
threshold = 0.5
neighboring_grid_visual = [['O' if cell >= threshold else 'X' for cell in row] for row in predicted_neighboring_grid]
for row in neighboring_grid_visual:
    print(" ".join(row))
print("True NeighborGrid:")
true_neighboring_grid_visual = [['O' if cell >= threshold else 'X' for cell in row] for row in true_neighboring_grid]
for row in true_neighboring_grid_visual:
    print(" ".join(row))