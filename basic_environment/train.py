import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from environments.GridEnvironmentMoving import CustomEnv
import os
from google.cloud import storage
from stable_baselines3 import DQN
import torch
from networks.CustomFeatureExtractor import CustomFeatureExtractor



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

    num_cpu = 8  # Number of processes to use

    # Create the environment
    env = CustomEnv()

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

    # Initialize PPO agent with new policy architecture
    model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="logs", device=device, buffer_size=50000, learning_rate=1e-6)

    # Train the agent
    model.learn(total_timesteps=int(40000))


    # Save the agent
    model.save("models/dqn")
