import subprocess
import csv
import cv2
from stable_baselines3 import DQN
import sys
sys.path.append("..")  # noqa: E402
from environments.GridEnvironment import GridEnvironment
import os
import logging
import time
from networks.CustomFeatureExtractor import CustomFeatureExtractor
import numpy as np

logging.basicConfig(level=logging.INFO)


# Set up the Bucket (google cloud storage)
# Define the bucket name
bucket_name = 'adlr_bucket'
model_directory = "basic_environment/models/DQN_0_0"

# Define the local download path
local_path = "models_from_bucket"

# Make the directory if it doesn't exist
if not os.path.exists(local_path):
    os.makedirs(local_path)

# Get the model filenames from the bucket using gsutil ls command
list_command = f"gsutil ls gs://{bucket_name}/{model_directory}"
output = subprocess.check_output(list_command, shell=True).decode('utf-8')
model_filenames = output.split("\n")[:-1]

# Integer sort the model filenames
model_filenames = sorted(model_filenames, key=lambda x: int(x.split("/")[-1].split(".")[0]))

# Define the filename for the most recent model
remote_filename = model_filenames[-1]

local_filename = remote_filename.split("/")[-1]

### For testing only: set local_filename to 500000.zip to reuse a downloaded model
# local_filename = "500000.zip"

# Download the model file if it doesn't already exist locally
if not os.path.exists(f"{local_path}/{local_filename}"):
    print(f"Downloading {remote_filename} from bucket {bucket_name} to {local_path}")

    # Get the total size of the file using gsutil du command
    du_command = f"gsutil du {remote_filename}"
    output = subprocess.check_output(du_command, shell=True).decode('utf-8')
    total_file_size = int(output.split()[0])  # Get the first part of the output, which is the size in bytes

    # Run the download command in the background
    download_command = f"gsutil -m cp -n -r {remote_filename} {local_path}/{local_filename} &"
    subprocess.run(download_command, shell=True, check=True)

    # Monitor the .gstmp file
    temp_filename = f"{local_path}/{local_filename}_.gstmp"
    while not os.path.exists(f"{local_path}/{local_filename}"):  # while the actual file has not been created
        if os.path.exists(temp_filename):  # if the temp file exists
            temp_filesize = os.path.getsize(temp_filename)  # get the current size of the file
            progress_percentage = (temp_filesize / total_file_size) * 100  # calculate progress as percentage
            logging.info(f"Current file size: {temp_filesize} bytes, Download progress: {progress_percentage:.2f}%")
        time.sleep(1)  # wait for a while before checking the size again

    print(f"\nDownloaded {remote_filename} from bucket {bucket_name} to {local_path}")
else:
    print(f"Model {local_filename} already exists in {local_path}")

# Load the model
custom_objects = {"lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0, "features_extractor_class": CustomFeatureExtractor}
model = DQN.load(f"{local_path}/{local_filename}", custom_objects=custom_objects, verbose=1)

# clear or create test_images directory
if os.path.exists("test_images"):
    for file in os.listdir("test_images"):
        os.remove(f"test_images/{file}")
else:
    os.mkdir("test_images")
    
#delete labels.csv if it exists
if os.path.exists("labels.csv"):
    os.remove("labels.csv")


# Define the field names for the CSV file
fieldnames = ['image_name', 'agent_pos_x', 'agent_pos_y', 'goal_pos_x', 'goal_pos_y'] + [f'neighbor_{i}_{j}' for i in range(7) for j in range(7)]

# Create the environment
env = GridEnvironment(num_obstacles=50)

# Reset the environment
obs, info = env.reset()

episode = 0
timestep = 0
while episode < 1000:

    # Open the CSV file for writing (append mode)
    with open('labels.csv', 'a', newline='') as csvfile:

        # randomly reset the environment sometimes after the first timestep (e.g. 10% of the time = 0.1)
        if np.random.random() < 1 and timestep == 1:
            obs, info = env.reset()
            terminated = False
            truncated = False
            episode += 1
            timestep = 0

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Extract the current frame info
        current_frame_info = env.get_current_frame_info()
        agent_pos = current_frame_info["agent_position"]
        goal_pos = current_frame_info["goal_position"]
        neighbors_content = current_frame_info["neighboring_cells_content"]

        # Extract the last RGB image
        image_name = f"episode_{episode}_timestep_{timestep}.png"
        last_rgb_image = obs[:, :, -3:]
        cv2.imwrite(f"test_images/{image_name}", last_rgb_image)

        # Write the data to the CSV file
        row_data = {
            'image_name': image_name,
            'agent_pos_x': agent_pos[0],
            'agent_pos_y': agent_pos[1],
            'goal_pos_x': goal_pos[0],
            'goal_pos_y': goal_pos[1]
        }
        row_data.update({f'neighbor_{i}_{j}': neighbors_content[i][j] for i in range(7) for j in range(7)})
        writer.writerow(row_data)

        # Predict the next action
        action, state = model.predict(obs, deterministic=True)
        # Take the action
        obs, reward, terminated, truncated, info = env.step(action)
        timestep += 1

        if terminated or truncated:

            #Print progress as percentage
            progress_percentage = (episode / 1000) * 100
            logging.info(f"Current episode: {episode}, Progress: {progress_percentage:.2f}%")

            # Reset the environment
            timestep = 0
            episode += 1
            obs, info = env.reset()

# Make sure to write the header only once at the beginning of the file (outside the loop)
with open('labels.csv', 'r') as file:
    content = file.read()
with open('labels.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()  # Write the header
    file.write(content)
