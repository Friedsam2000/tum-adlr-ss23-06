import sys
import numpy as np
import th as th
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import DQNPolicy

sys.path.append("..")  # noqa: E402
from environments.GridEnvironment import GridEnvironment
import os
from google.cloud import storage
from stable_baselines3 import DQN
import torch
import torch.nn as nn
import gymnasium as gym
import sys
sys.path.append("..")  # noqa: E402
from supervisedExtractor.cnnExtractor import CNNExtractor


import torch.nn.functional as F

class CustomMLPExtractor(nn.Module):
    def __init__(self, input_size):
        super(CustomMLPExtractor, self).__init__()

        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return x



class PretrainedFeaturesExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
    """

    def __init__(self, observation_space: gym.spaces.Box,
                 features_dim: int = 4 * 4):  # update with your actual features_dim
        super(PretrainedFeaturesExtractor, self).__init__(observation_space, features_dim)

        # Load the model
        self.pretrained_model = CNNExtractor()
        model_path = 'model.pth'
        self.pretrained_model.load_state_dict(torch.load(model_path, map_location=device))
        self.pretrained_model.to(device)
        self.pretrained_model.eval()

        # Disable gradient computation for the pretrained model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Adjusted the method to accept [batch_size, num_frames*channels, width, height]
        batch_size, _, _, _ = observations.shape
        num_frames = 4  # Update if different
        features = []
        for frame in range(num_frames):
            single_frame_observation = observations[:, frame * 3:(frame + 1) * 3, :,
                                       :]  # shape: [batch_size, channels, width, height]
            x_grid, x_position = self.pretrained_model(single_frame_observation)
            features.append(x_position)  # concatenate the output along the feature dimension
        return torch.cat(features, dim=1)  # concatenate all frames along the feature dimension


def make_env(rank):
    def _init():
        env = GridEnvironment()
        return env

    return _init


if __name__ == "__main__":

    print("GPU is available: ")
    print(torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bucket_name = 'adlr_bucket'
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    num_cpu = 1  # Number of processes to use
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    env = VecMonitor(env)

    if not os.path.exists("logs"):
        os.makedirs("logs")

    if not os.path.exists("models"):
        os.makedirs("models")

    logs_folders = os.listdir("logs")

    # Define the policy kwargs
    policy_kwargs = dict(
        features_extractor_class=PretrainedFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=4 * 4)  # or the dimensionality of your pretrained model output
    )


    class CustomPolicy(DQNPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomPolicy, self).__init__(*args, **kwargs)

            self.mlp = CustomMLPExtractor(self.features_dim)

        def extract_features(self, obs: th.Tensor) -> th.Tensor:
            preprocessed_obs = super().extract_features(obs)
            return self.mlp(preprocessed_obs)


    model = DQN(CustomPolicy, env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="logs", device=device,
                buffer_size=5000, learning_starts=5000)

    if not os.path.exists(f"models/DQN_{len(logs_folders)}_0"):
        os.makedirs(f"models/DQN_{len(logs_folders)}_0")

    best_reward = -np.inf

    TIMESTEPS_PER_SAVE = 5000
    MAX_TIMESTEPS = 100000000
    while model.num_timesteps < MAX_TIMESTEPS:
        model.learn(total_timesteps=TIMESTEPS_PER_SAVE, reset_num_timesteps=False,
                    tb_log_name=f"DQN_{len(logs_folders)}", log_interval=100)

        #display network architecture
        print(model.policy)



        reward_mean = np.mean([ep['r'] for ep in list(model.ep_info_buffer)[-100:]])

        if reward_mean > best_reward:
            best_reward = reward_mean
            print(f"Saving model with new best reward mean {reward_mean}")
            model.save(f"models/DQN_{len(logs_folders)}_0/{model.num_timesteps}")

            blob = bucket.blob(f"basic_environment/pretrainedExtractor/models/DQN_{len(logs_folders)}_0/{model.num_timesteps}.zip")
            blob.upload_from_filename(f"models/DQN_{len(logs_folders)}_0/{model.num_timesteps}.zip")
            print(f"Uploaded model {model.num_timesteps}.zip to bucket")

            os.remove(f"models/DQN_{len(logs_folders)}_0/{model.num_timesteps}.zip")

        logs = os.listdir(f"logs/DQN_{len(logs_folders)}_0")
        logs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        latest_log = logs[-1]

        blob = bucket.blob(f"basic_environment/pretrainedExtractor/logs/DQN_{len(logs_folders)}_0/{latest_log}")
        blob.upload_from_filename(f"logs/DQN_{len(logs_folders)}_0/{latest_log}")
