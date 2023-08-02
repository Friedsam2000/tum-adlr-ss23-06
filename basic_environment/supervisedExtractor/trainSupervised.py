import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from cnnExtractor import CNNExtractor
from dataPreprocessor import load_data
from torch.utils.tensorboard import SummaryWriter
import numpy as np


# Define a custom loss function to handle class imbalance
def custom_loss(predictions, labels):
    agent_goal_labels = labels[:, :4]
    grid_labels = labels[:, 4:]

    agent_goal_predictions = predictions[:, :4]
    grid_predictions = predictions[:, 4:]  # No sigmoid here, as BCEWithLogitsLoss includes it

    loss_position = nn.MSELoss()(agent_goal_predictions, agent_goal_labels)

    # Weight for the positive class (obstacles), increased to 100
    pos_weight = torch.full_like(grid_labels, 1)

    # Compute the binary cross-entropy loss using the weighted positive class
    loss_grid = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(grid_predictions, grid_labels)

    return loss_position + loss_grid


# Number of images to train on
num_images = 1000

# Define transformation for the images
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(script_dir, '../img_data_generation/labels.csv')
images_dir_path = os.path.join(script_dir, '../img_data_generation')
dataset = load_data(csv_file=csv_file_path, images_dir=images_dir_path, transform=transform)

# Take a subset of the dataset if the size is more than num_images
if len(dataset) > num_images:
    dataset, _ = torch.utils.data.random_split(dataset, [num_images, len(dataset) - num_images])

# Split the data into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Instantiate the model
model = CNNExtractor()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create a SummaryWriter to log data to TensorBoard
writer = SummaryWriter()

# Training loop
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image']
        labels = batch['label'].clone().detach().float()
        optimizer.zero_grad()
        predictions = model(images)
        combined_loss = custom_loss(predictions, labels)
        combined_loss.backward()
        optimizer.step()
        train_loss += combined_loss.item()
        writer.add_scalar('train_loss', combined_loss.item(), epoch * len(train_loader) + batch_idx)

    # Evaluate on validation set
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image']
            labels = batch['label'].clone().detach().float()
            predictions = model(images)
            combined_loss = custom_loss(predictions, labels)
            val_loss += combined_loss.item()

    # Log training and validation losses to TensorBoard every epoch
    writer.add_scalar('epoch_train_loss', train_loss / len(train_loader), epoch)
    writer.add_scalar('epoch_val_loss', val_loss / len(val_loader), epoch)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"  Train loss: {train_loss / len(train_loader)}")
    print(f"  Validation loss: {val_loss / len(val_loader)}")

# Save the model
torch.save(model.state_dict(), 'model.pth')