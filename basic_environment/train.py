from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from GridEnvironment import CustomEnv
import os
import torch
from google.cloud import storage


def make_env(grid_size, rank):
    def _init():
        env = CustomEnv(grid_size=grid_size)
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
    grid_size = (8, 8)

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
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="logs", device=device, learning_rate=0.0001)

    # Train agent
    TIMESTEPS_PER_SAVE = 50000
    MAX_TIMESTEPS = 3000000
    while model.num_timesteps < MAX_TIMESTEPS:
        model.learn(total_timesteps=TIMESTEPS_PER_SAVE, reset_num_timesteps=False,
                    tb_log_name=f"PPO_{len(logs_folders)}")
        # create the folder for the model
        if not os.path.exists(f"models/PPO_{len(logs_folders)}_0"):
            os.makedirs(f"models/PPO_{len(logs_folders)}_0")
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
