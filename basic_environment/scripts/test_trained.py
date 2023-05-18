from stable_baselines3 import PPO
from GridEnvironment import CustomEnv
import os
import google.cloud.storage
import shutil


# Set up the Bucket
# Define the bucket name
bucket_name = 'adlr_bucket'
# Initialize a storage client
storage_client = google.cloud.storage.Client()
# Get the bucket object
bucket = storage_client.get_bucket(bucket_name)

# Get all model filenames from the bucket
PPO_Iteration = "PPO_0_0"
blobs = bucket.list_blobs(prefix=f"basic_environment/models/{PPO_Iteration}")
model_filenames = []
for blob in blobs:
    model_filenames.append(blob.name)

# Integer sort the model filenames
model_filenames = sorted(model_filenames, key=lambda x: int(x.split("/")[-1].split(".")[0]))

# Empy or create the models_from_bucket directory using shutil
if os.path.exists("models_from_bucket"):
    shutil.rmtree("models_from_bucket")
os.makedirs("models_from_bucket")



# Download the model with the highest number of steps in the models_from_bucket directory
model_filename = model_filenames[-1]
blob = bucket.blob(model_filename)
blob.download_to_filename(f"models_from_bucket/" + model_filename.split("/")[-1])
print(f"Downloaded {model_filename} from bucket {bucket_name} to models_from_bucket directory")

# Load the model
custom_objects = {"lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0}
model = PPO.load(f"models_from_bucket/" + model_filename.split("/")[-1], custom_objects=custom_objects, verbose=1)
print(f"Loaded {model_filename} from models_from_bucket directory")

# Create the environment
env = CustomEnv(grid_size=(18, 18), draw_num_old_agent_pos=3)

# Test the model
obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()