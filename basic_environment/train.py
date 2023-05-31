import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3 import PPO
from environments.GridEnvironment import CustomEnv
from environments.GridEnvironment import CustomEnv_rc
from environments.GridEnvironment import CustomEnv_sr_fp
import os
import torch
from google.cloud import storage



def make_env(grid_size, rank):
    def _init():
        env = CustomEnv_sr_fp(grid_size=grid_size)
        return env

    return _init


if __name__ == "__main__":
    
    PPO_Iteration = "CNN_3"
    PPO_Policy = "CNN"
    print(PPO_Iteration)
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
    n_steps = 256
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="logs", device=device, n_steps=n_steps, batch_size=512*8)

    # create the folder for the model
    if not os.path.exists(f"models/PPO_{PPO_Iteration}"):
        os.makedirs(f"models/PPO_{PPO_Iteration}")

    best_reward = -np.inf
    log_save_counter = 0

    # Train agent
    TIMESTEPS_PER_SAVE = n_steps*num_cpu*20
    MAX_TIMESTEPS = 7500000
    while model.num_timesteps < MAX_TIMESTEPS:
        model.learn(total_timesteps=TIMESTEPS_PER_SAVE, reset_num_timesteps=False,
                    tb_log_name=f"PPO_{PPO_Policy}")

        # get the mean reward of the last 100 episodes
        reward_mean = np.mean([ep['r'] for ep in list(model.ep_info_buffer)[-100:]])

        # if the reward mean is better than the best reward, save the model
        if reward_mean > best_reward:
            best_reward = reward_mean
            print(f"Saving model with new best reward mean {reward_mean}")
            model.save(f"models/PPO_{PPO_Iteration}/{model.num_timesteps}")

            # upload the model to the bucket
            blob = bucket.blob(f"data_Matthias/models/PPO_{PPO_Iteration}/{model.num_timesteps}.zip")
            blob.upload_from_filename(f"models/PPO_{PPO_Iteration}/{model.num_timesteps}.zip")
            print(f"Uploaded model {model.num_timesteps}.zip to bucket")
        if log_save_counter%2 == 0:
            # get the latest log file
            logs = os.listdir(f"logs/PPO_{PPO_Policy}_0")
            logs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            latest_log = logs[-1]
            # upload the new log file to the bucket
            blob = bucket.blob(f"data_Matthias/logs/PPO_{PPO_Iteration}/{latest_log}")
            blob.upload_from_filename(f"logs/PPO_{PPO_Policy}_0/{latest_log}")
        
        log_save_counter = log_save_counter + 1
