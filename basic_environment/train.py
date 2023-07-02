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
    model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="logs", device=device, learning_rate=3e-5, buffer_size=10000, learning_starts=10000)



    # create the folder for the model
    if not os.path.exists(f"models/DQN_{len(logs_folders)}_0"):
        os.makedirs(f"models/DQN_{len(logs_folders)}_0")

    best_reward = -np.inf

    # Train agent
    TIMESTEPS_PER_SAVE = 10000
    MAX_TIMESTEPS = 100000000
    while model.num_timesteps < MAX_TIMESTEPS:
        model.learn(total_timesteps=TIMESTEPS_PER_SAVE, reset_num_timesteps=False,
                    tb_log_name=f"DQN_{len(logs_folders)}", log_interval = 50)

        # get the mean reward of the last 10 episodes
        reward_mean = np.mean([ep['r'] for ep in list(model.ep_info_buffer)[-100:]])

        # if the reward mean is better than the best reward, save the model
        if reward_mean > best_reward:
            best_reward = reward_mean
            print(f"Saving model with new best reward mean {reward_mean}")
            model.save(f"models/DQN_{len(logs_folders)}_0/{model.num_timesteps}")

            # upload the model to the bucket
            blob = bucket.blob(f"basic_environment/models/DQN_{len(logs_folders)}_0/{model.num_timesteps}.zip")
            blob.upload_from_filename(f"models/DQN_{len(logs_folders)}_0/{model.num_timesteps}.zip")
            print(f"Uploaded model {model.num_timesteps}.zip to bucket")

            # delete the model locally
            os.remove(f"models/DQN_{len(logs_folders)}_0/{model.num_timesteps}.zip")

        # get the latest log file
        logs = os.listdir(f"logs/DQN_{len(logs_folders)}_0")
        logs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        latest_log = logs[-1]
        # upload the new log file to the bucket
        blob = bucket.blob(f"basic_environment/logs/DQN_{len(logs_folders)}_0/{latest_log}")
        blob.upload_from_filename(f"logs/DQN_{len(logs_folders)}_0/{latest_log}")
