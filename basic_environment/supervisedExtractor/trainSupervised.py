import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cnnExtractor import CNNExtractor
from dataPreprocessor import load_data
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset
from google.cloud import storage
from datetime import datetime

# Check for CUDA availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device} device')


# Define a custom loss function for grid prediction
def custom_loss(predictions_grid, predictions_pos, grid_labels, pos_labels):
    predictions_grid = predictions_grid.view(predictions_grid.size(0),
                                             -1)  # Reshaping predictions to match the target size
    pos_weight = torch.full_like(grid_labels, 1)
    loss_grid = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(predictions_grid, grid_labels)

    # MSE Loss for positions
    loss_pos = nn.MSELoss()(predictions_pos, pos_labels)

    # You can adjust the ratio of grid to position loss by using a different weight
    grid_loss_weight = 1
    pos_loss_weight = 0.1

    return grid_loss_weight * loss_grid + pos_loss_weight * loss_pos


# Load the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(script_dir, '../imageDataGeneration/labels.csv')
images_dir_path = os.path.join(script_dir, '../imageDataGeneration')
dataset = load_data(csv_file=csv_file_path, images_dir=images_dir_path)

# Limit the dataset to the first 10000 samples
max_images = 500000
dataset = Subset(dataset, indices=range(max_images))

# Split the data into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders with multiple workers
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Instantiate the model
model = CNNExtractor()
model.to(device)  # Move the model to the device

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Google Cloud Storage configurations
bucket_name = 'adlr_bucket'
log_dir = 'supervised/logs'
model_dir = 'supervised/model'
storage_client = storage.Client()
bucket = storage_client.get_bucket(bucket_name)  # Define the bucket here

# Save TensorBoard logs to a new directory
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
tb_log_dir = f'{log_dir}/{current_time}'
writer = SummaryWriter(tb_log_dir)

# Initially set min_val_loss to a very high value
min_val_loss = float('inf')

# Before the training loop
previous_model_path = None

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        grid_labels = batch['label'][:, 4:].clone().detach().float().to(
            device)  # Only obstacle grid, ignoring first 4 elements
        pos_labels = batch['label'][:, :4].clone().detach().float().to(device)  # Only the positions
        optimizer.zero_grad()
        predictions_grid, predictions_pos = model(images)
        combined_loss = custom_loss(predictions_grid, predictions_pos, grid_labels, pos_labels)
        combined_loss.backward()
        optimizer.step()
        train_loss += combined_loss.item()
        writer.add_scalar('train_loss', combined_loss.item(), epoch * len(train_loader) + batch_idx)

    # Evaluate on validation set
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            grid_labels = batch['label'][:, 4:].clone().detach().float().to(
                device)  # Only obstacle grid, ignoring first 4 elements
            pos_labels = batch['label'][:, :4].clone().detach().float().to(device)  # Only the positions
            predictions_grid, predictions_pos = model(images)
            combined_loss = custom_loss(predictions_grid, predictions_pos, grid_labels, pos_labels)
            val_loss += combined_loss.item()

    # Log training and validation losses to TensorBoard every epoch
    writer.add_scalar('epoch_train_loss', train_loss / len(train_loader), epoch)
    writer.add_scalar('epoch_val_loss', val_loss / len(val_loader), epoch)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"  Train loss: {train_loss / len(train_loader)}")
    print(f"  Validation loss: {val_loss / len(val_loader)}")

    # Within the training loop, in the validation loss improvement check
    if val_loss / len(val_loader) < min_val_loss:
        min_val_loss = val_loss / len(val_loader)

        # Save the model with a new name
        model_path_local = f'{current_time}_model.pth'
        torch.save(model.state_dict(), model_path_local)

        # Upload the model to Google Cloud Storage
        blob = bucket.blob(f'basic_environment/{model_dir}/{current_time}_model.pth')
        blob.upload_from_filename(model_path_local)
        print(f"New lowest validation loss, model saved and uploaded to {blob.public_url}")

        # Remove the previous saved model if it exists
        if previous_model_path is not None and os.path.exists(previous_model_path):
            os.remove(previous_model_path)

        # Update the previous model path
        previous_model_path = model_path_local

    # Upload TensorBoard logs to Google Cloud Storage every epoch
    try:
        log_files = os.listdir(tb_log_dir)
        for file in log_files:
            file_path = os.path.join(tb_log_dir, file)
            if os.path.isfile(file_path):  # Ignore subdirectories
                blob = bucket.blob(f'basic_environment/{log_dir}/{current_time}/{file}')
                blob.upload_from_filename(file_path)
        print(f"Successfully uploaded logs to gs://{bucket_name}/{log_dir}/{current_time}/")
    except Exception as e:
        print(f"Failed to upload logs: {e}")

# Close the TensorBoard writer
writer.close()

# Delete local TensorBoard logs
os.system(f'rm -rf {tb_log_dir}')
print(f"Successfully deleted local TensorBoard logs {tb_log_dir}")
