import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
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
def custom_loss(predictions, grid_labels):
    predictions = predictions.view(predictions.size(0), -1)  # Reshaping predictions to match the target size
    pos_weight = torch.full_like(grid_labels, 2)

    loss_grid = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(predictions, grid_labels)
    return loss_grid

# Define transformation for the images
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(script_dir, '../img_data_generation/labels.csv')
images_dir_path = os.path.join(script_dir, '../img_data_generation')
dataset = load_data(csv_file=csv_file_path, images_dir=images_dir_path, transform=transform)

# Limit the dataset to the first 10000 samples
max_images = 10000
dataset = Subset(dataset, indices=range(max_images))

# Split the data into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders with multiple workers
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

# Instantiate the model
model = CNNExtractor()
model.to(device) # Move the model to the device

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Google Cloud Storage configurations
bucket_name = 'adlr_bucket'
log_dir = 'supervised/logs'
model_dir = 'model'
storage_client = storage.Client()

# Save TensorBoard logs to a new directory
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
tb_log_dir = f'{log_dir}/{current_time}'
writer = SummaryWriter(tb_log_dir)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        grid_labels = batch['label'][:, 4:].clone().detach().float().to(device) # Only obstacle grid, ignoring first 4 elements
        optimizer.zero_grad()
        predictions = model(images)
        combined_loss = custom_loss(predictions, grid_labels)
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
            grid_labels = batch['label'][:, 4:].clone().detach().float().to(device) # Only obstacle grid, ignoring first 4 elements
            predictions = model(images)
            combined_loss = custom_loss(predictions, grid_labels)
            val_loss += combined_loss.item()

    # Log training and validation losses to TensorBoard every epoch
    writer.add_scalar('epoch_train_loss', train_loss / len(train_loader), epoch)
    writer.add_scalar('epoch_val_loss', val_loss / len(val_loader), epoch)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"  Train loss: {train_loss / len(train_loader)}")
    print(f"  Validation loss: {val_loss / len(val_loader)}")

    # Upload TensorBoard logs to Google Cloud Storage every epoch
    try:
        bucket = storage_client.get_bucket(bucket_name)
        log_files = os.listdir(tb_log_dir)
        for file in log_files:
            file_path = os.path.join(tb_log_dir, file)
            if os.path.isfile(file_path):  # Ignore subdirectories
                blob = bucket.blob(f'{log_dir}/{current_time}/{file}')
                blob.upload_from_filename(file_path)
        print(f"Successfully uploaded logs to gs://{bucket_name}/{log_dir}/{current_time}/")
    except Exception as e:
        print(f"Failed to upload logs: {e}")

# Save the model with a new name
model_path_local = f'{current_time}_model.pth'
torch.save(model.state_dict(), model_path_local)

# Upload the model to Google Cloud Storage
blob = bucket.blob(f'{model_dir}/{current_time}_model.pth')
blob.upload_from_filename(model_path_local)
print(f"Successfully uploaded model to {blob.public_url}")
