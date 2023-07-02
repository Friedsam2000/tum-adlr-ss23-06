import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from environments.GridEnvironmentMoving import CustomEnv
import os
from google.cloud import storage
from stable_baselines3 import DQN
import torch
from networks.CustomFeatureExtractor import CustomFeatureExtractor


def make_env(rank):
    def _init():
        env = CustomEnv()
        return env

    return _init


if __name__ == "__main__":

    # Set up the GPU or use the CPU
    print("GPU is available: ")
    print(torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set up the bucket (google cloud storage)
    # Define the bucket name
    bucket_name = 'adlr_bucket'
    # Initialize a storage client
    storage_client = storage.Client()
    # Get the bucket object
    bucket = storage_client.get_bucket(bucket_name)

    num_cpu = 1  # Number of processes to use

    # Create the vectorized environment
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    # add a monitor wrapper
    env = VecMonitor(env)

    # Create logs if not existing
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Create models if not existing
    if not os.path.exists("models"):
        os.makedirs("models")

    # Check how many folders are in logs
    logs_folders = os.listdir("logs")

    # Define the policy kwargs
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
    )

    # Initialize DQN agent with new policy architecture
    model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="logs", device=device,
                learning_rate=3e-5, buffer_size=50000)

    # Train agent
    model.learn(total_timesteps=int(1e5))

    # Save the agent
    model.save("models/DQN")

    # upload the model to the bucket
    blob = bucket.blob(f"basic_environment/models/DQN.zip")
    blob.upload_from_filename(f"models/DQN.zip")
    print(f"Uploaded model DQN.zip to bucket")
    # delete the model locally
    os.remove(f"models/DQN.zip")


