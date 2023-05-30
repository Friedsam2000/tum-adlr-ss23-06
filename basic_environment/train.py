import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3 import PPO
from environments.GridEnvironment import CustomEnv
import os
import torch
from google.cloud import storage
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym

class CustomCnn(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCnn, self).__init__(observation_space, features_dim)

        # Assume input has shape (48, 48, 3)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=CustomCnn,
    features_extractor_kwargs=dict(features_dim=256),
)



def make_env(grid_size, rank):
    def _init():
        env = CustomEnv(grid_size=grid_size, num_last_agent_pos=100)
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

    num_cpu = 16  # Number of processes to use
    grid_size = (16, 16)

    # Create the vectorized environment
    env = SubprocVecEnv([make_env(grid_size, i) for i in range(num_cpu)])
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

    # Initialize PPO agent with CNN policy
    n_steps = 64
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="logs", device=device,
                n_steps=n_steps)

    # create the folder for the model
    if not os.path.exists(f"models/PPO_{len(logs_folders)}_0"):
        os.makedirs(f"models/PPO_{len(logs_folders)}_0")

    best_reward = -np.inf

    # Train agent
    TIMESTEPS_PER_SAVE = 16384
    MAX_TIMESTEPS = 100000000
    while model.num_timesteps < MAX_TIMESTEPS:
        model.learn(total_timesteps=TIMESTEPS_PER_SAVE, reset_num_timesteps=False,
                    tb_log_name=f"PPO_{len(logs_folders)}")

        # get the mean reward of the last 100 episodes
        reward_mean = np.mean([ep['r'] for ep in list(model.ep_info_buffer)[-100:]])

        # if the reward mean is better than the best reward, save the model
        if reward_mean > best_reward:
            best_reward = reward_mean
            print(f"Saving model with new best reward mean {reward_mean}")
            model.save(f"models/PPO_{len(logs_folders)}_0/{model.num_timesteps}")

            # upload the model to the bucket
            blob = bucket.blob(f"basic_environment/models/PPO_{len(logs_folders)}_0/{model.num_timesteps}.zip")
            blob.upload_from_filename(f"models/PPO_{len(logs_folders)}_0/{model.num_timesteps}.zip")
            print(f"Uploaded model {model.num_timesteps}.zip to bucket")


        # get the latest log file
        logs = os.listdir(f"logs/PPO_{len(logs_folders)}_0")
        logs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        latest_log = logs[-1]
        # upload the new log file to the bucket
        blob = bucket.blob(f"basic_environment/logs/PPO_{len(logs_folders)}_0/{latest_log}")
        blob.upload_from_filename(f"logs/PPO_{len(logs_folders)}_0/{latest_log}")

