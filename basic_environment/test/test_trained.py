from stable_baselines3 import PPO
from environments.GridEnvironmentMoving import CustomEnv
import os
import google.cloud.storage
import shutil
from networks.CustomFeatureExtractor import CustomFeatureExtractor

# Set up the Bucket (google cloud storage)
# Define the bucket name
bucket_name = 'adlr_bucket'
# Initialize a storage client
storage_client = google.cloud.storage.Client()
# Get the bucket object
bucket = storage_client.get_bucket(bucket_name)

# Get all model filenames from the bucket
PPO_Iteration = "PPO_38_0"
blobs = bucket.list_blobs(prefix=f"basic_environment/models/{PPO_Iteration}")
model_filenames = []
for blob in blobs:
    model_filenames.append(blob.name)

# Integer sort the model filenames
model_filenames = sorted(model_filenames, key=lambda x: int(x.split("/")[-1].split(".")[0]))

# Check if the models_from_bucket directory exists
if not os.path.exists("models_from_bucket"):
    os.makedirs("models_from_bucket")

# Check if the model is downloaded
if not os.path.exists(f"models_from_bucket/{model_filenames[-1].split('/')[-1]}"):
    print(f"Downloading {model_filenames[-1]} from bucket {bucket_name} to models_from_bucket directory")
    # Download the model with the highest number of steps in the models_from_bucket directory
    model_filename = model_filenames[-1]
    blob = bucket.blob(model_filename)
    blob.download_to_filename(f"models_from_bucket/" + model_filename.split("/")[-1])
    print(f"Downloaded {model_filename} from bucket {bucket_name} to models_from_bucket directory")
else:
    print(f"Model {model_filenames[-1]} already exists in models_from_bucket directory")
    model_filename = model_filenames[-1]

# Load the model
custom_objects = {"lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0, "features_extractor_class": CustomFeatureExtractor}
model = PPO.load(f"models_from_bucket/" + model_filename.split("/")[-1], custom_objects=custom_objects, verbose=1)

# Create the environment
env = CustomEnv()

# Test the model
obs, info = env.reset()
goals_reached = 0
obstacles_hit = 0
timeouts = 0
episodes = 0
while episodes < 500:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    env.render()
    if terminated:
        if reward == 1:
            goals_reached += 1
        elif reward == -1:
            obstacles_hit += 1

    if truncated:
        timeouts += 1

    if terminated or truncated:
        episodes += 1
        obs, info = env.reset()



print(f"Succes rate: {goals_reached / episodes}")
print(f"Obstacles hit: {obstacles_hit / episodes}")
print(f"Timeouts: {timeouts / episodes}")