import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3 import PPO
from environments.ModifiedGridEnvironment import ModifiedCustomEnv
import os
import torch
from google.cloud import storage
import shutil



def make_env(grid_size, rank):
    def _init():
        env = ModifiedCustomEnv(grid_size=grid_size)
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
    if not os.path.exists("logs_modified"):
        os.makedirs("logs_modified")

    # Create models if not existing
    if not os.path.exists("models_modified"):
        os.makedirs("models_modified")

    # Check how many folders are in logs
    logs_folders = os.listdir("logs_modified")

    # Get all model filenames from the bucket
    PPO_Iteration = "PPO_0_0"
    blobs = bucket.list_blobs(prefix=f"basic_environment/models/{PPO_Iteration}")
    model_filenames = []
    for blob in blobs:
        model_filenames.append(blob.name)

    # Integer sort the model filenames
    model_filenames = sorted(model_filenames, key=lambda x: int(x.split("/")[-1].split(".")[0]))

    # Empty or create the models_from_bucket directory using shutil
    if os.path.exists("models_to_modify"):
        shutil.rmtree("models_to_modify")
    os.makedirs("models_to_modify")

    # Download the model with the highest number of steps in the models_from_bucket directory
    model_filename = model_filenames[-1]
    blob = bucket.blob(model_filename)
    blob.download_to_filename(f"models_to_modify/" + model_filename.split("/")[-1])
    print(f"Downloaded {model_filename} from bucket {bucket_name} to models_to_modify directory")

    # Load the model
    custom_objects = {"lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0}
    env.reset()
    model = PPO.load(f"models_to_modify/" + model_filename.split("/")[-1], env=env, custom_objects=custom_objects, verbose=1)
    # set new tensorboard log directory
    model.tensorboard_log = "logs_modified"
    print(f"Loaded {model_filename} from models_to_modify directory")

    # create the folder for the model
    if not os.path.exists(f"models_modified/PPO_{len(logs_folders)}"):
        os.makedirs(f"models_modified/PPO_{len(logs_folders)}")

    best_reward = -np.inf

    # Continue training
    TIMESTEPS_PER_SAVE = model.n_steps * num_cpu
    MAX_TIMESTEPS = 3000000
    while model.num_timesteps < MAX_TIMESTEPS:
        model.learn(total_timesteps=TIMESTEPS_PER_SAVE, reset_num_timesteps=False)

        # get the mean reward of the last 100 episodes
        reward_mean = np.mean([ep['r'] for ep in list(model.ep_info_buffer)[-100:]])

        # if the reward mean is better than the best reward, save the model
        if reward_mean > best_reward:
            best_reward = reward_mean
            print(f"Saving model with new best reward mean {reward_mean}")
            model.save(f"models_modified/PPO_{len(logs_folders)}/{model.num_timesteps}")

            # upload the model to the bucket
            blob = bucket.blob(f"basic_environment/models/PPO_{len(logs_folders)}_0_modified/{model.num_timesteps}.zip")
            blob.upload_from_filename(f"models_modified/PPO_{len(logs_folders)}/{model.num_timesteps}.zip")
            print(f"Uploaded model {model.num_timesteps}.zip to bucket")

        # get the latest log file
        logs = os.listdir(f"logs_modified/PPO_{len(logs_folders)}")
        logs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        latest_log = logs[-1]
        # upload the new log file to the bucket
        blob = bucket.blob(f"basic_environment/logs/PPO_{len(logs_folders)}_0_modified/{latest_log}")
        blob.upload_from_filename(f"logs_modified/PPO_{len(logs_folders)}/{latest_log}")
