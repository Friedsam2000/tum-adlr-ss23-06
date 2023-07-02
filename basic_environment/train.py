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
    print("GPU is available: ")
    print(torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bucket_name = 'adlr_bucket'
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    num_cpu = 8
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    env = VecMonitor(env)

    if not os.path.exists("logs"):
        os.makedirs("logs")
    if not os.path.exists("models"):
        os.makedirs("models")
    logs_folders = os.listdir("logs")

    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
    )

    model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="logs", device=device,
                learning_rate=3e-5, buffer_size=int(1e4))

    if not os.path.exists(f"models/DQN_{len(logs_folders)}_0"):
        os.makedirs(f"models/DQN_{len(logs_folders)}_0")

    best_reward = -np.inf
    TIMESTEPS_PER_SAVE = 16384
    MAX_TIMESTEPS = 100000000

    while model.num_timesteps < MAX_TIMESTEPS:
        # Here we do not call model.learn() to learn every timestep. Instead we will call model.train() and model.update_target_net()
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)

            # Store transition in the replay buffer.
            model.replay_buffer.add(obs, action, rewards, done, info)

            if model.replay_buffer.size() > model.batch_size:
                model.train(replay_data=model.replay_buffer.sample(model.batch_size, env=env))
                if model.num_timesteps % model.target_update_interval == 0:
                    model.update_target_net()

        reward_mean = np.mean([ep['r'] for ep in list(model.ep_info_buffer)[-100:]])
        if reward_mean > best_reward:
            best_reward = reward_mean
            print(f"Saving model with new best reward mean {reward_mean}")
            model.save(f"models/DQN_{len(logs_folders)}_0/{model.num_timesteps}")
            blob = bucket.blob(f"basic_environment/models/DQN_{len(logs_folders)}_0/{model.num_timesteps}.zip")
            blob.upload_from_filename(f"models/DQN_{len(logs_folders)}_0/{model.num_timesteps}.zip")
            print(f"Uploaded model {model.num_timesteps}.zip to bucket")
            os.remove(f"models/DQN_{len(logs_folders)}_0/{model.num_timesteps}.zip")

        logs = os.listdir(f"logs/DQN_{len(logs_folders)}_0")
        logs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        latest_log = logs[-1]
        blob = bucket.blob(f"basic_environment/logs/DQN_{len(logs_folders)}_0/{latest_log}")
        blob.upload_from_filename(f"logs/DQN_{len(logs_folders)}_0/{latest_log}")
