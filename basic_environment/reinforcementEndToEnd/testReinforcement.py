import subprocess
from stable_baselines3 import DQN
from environments.GridEnvironment import GridEnvironment
import os
import logging
import time
from CustomFeatureExtractor import CustomFeatureExtractor

logging.basicConfig(level=logging.INFO)


# Set up the Bucket (google cloud storage)
# Define the bucket name
bucket_name = 'adlr_bucket'
model_directory = "basic_environment/reinforcementEndToEnd/DQN_0_0"

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
    print(f"Model {remote_filename} already exists in {local_path}")



# Load the model
custom_objects = {"lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0, "features_extractor_class": CustomFeatureExtractor}
model = DQN.load(f"{local_path}/{local_filename}", custom_objects=custom_objects, verbose=1)

# Rest of the code remains the same

# Create the environment
env = GridEnvironment()

# Print the network architecture
print(model.policy)

# Test the model
obs, info = env.reset()
goals_reached = 0
obstacles_hit = 0
timeouts = 0
episodes = 0
# Print testing
num_episodes = 500
print("Testing the model")
while episodes < num_episodes:



    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # env.render()
    if terminated:
        if reward == 1:
            goals_reached += 1
        elif reward == -1:
            obstacles_hit += 1

    if truncated:
        timeouts += 1

    if terminated or truncated:

        # Print progress in %
        if episodes % 10 == 0:
            print(f"{episodes / num_episodes * 100}%")

        episodes += 1
        obs, info = env.reset()



print(f"Succes rate: {goals_reached / episodes}")
print(f"Obstacles hit: {obstacles_hit / episodes}")
print(f"Timeouts: {timeouts / episodes}")