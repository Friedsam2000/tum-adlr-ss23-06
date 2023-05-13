import os
import gym
from stable_baselines3 import PPO
from google.cloud import storage
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env


# Check if GPU is available
print("GPU is available: ")
print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"


# Define the bucket name
bucket_name = 'adlr_bucket'

# Initialize a storage client
storage_client = storage.Client()

# Get the bucket object
bucket = storage_client.get_bucket(bucket_name)

num_cpu = 16  # Adjust according to your CPU
env = make_vec_env('LunarLander-v2', n_envs=num_cpu)


models_dir = "models/PPO_0"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


def getLatestModel(dir=models_dir):
    if os.path.exists(dir) and len(os.listdir(dir)) > 0:
        models = os.listdir(dir)
        # integer sort
        models.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        return models[-1]
    else:
        return None


# Start training or continue training at the last model
TIMESTEPS = 10000
MAX_TIMESTEPS = 1000000
latest_model = getLatestModel()
if latest_model is not None:
    model = PPO.load(f"{models_dir}/{latest_model}", env=env, verbose=1, tensorboard_log=logdir)
    print(f"Continue training at timestep {model.num_timesteps}")
else:
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir, device=device)
    print("Start training from scratch")
while model.num_timesteps < MAX_TIMESTEPS:
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model_name = f"{models_dir}/{model.num_timesteps}"
    model.save(model_name)

    # upload the new model to the bucket
    blob = bucket.blob(f"lunar_landar/{model_name}.zip")
    blob.upload_from_filename(f"{model_name}.zip")

    # get the latest log file
    logs = os.listdir(f"{logdir}/PPO_0")
    logs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    latest_log = logs[-1]
    # upload the new log file to the bucket
    blob = bucket.blob(f"lunar_landar/logs/PPO_0/{latest_log}")
    blob.upload_from_filename(f"{logdir}/PPO_0/{latest_log}")

# Test the latest model locally
# latest_model = getLatestModel()
# if latest_model is not None:
#     model = PPO.load(f"{models_dir}/{latest_model}", env=env, verbose=1, tensorboard_log=logdir)
#     print(f"Testing model at timestep {model.num_timesteps}")
#     obs = env.reset()
#     for i in range(1000):
#         action, _states = model.predict(obs, deterministic=True)
#         obs, rewards, dones, info = env.step(action)
#         env.render()
#         if dones:
#             break
#     env.close()

# Download all models from the bucket
# blobs = bucket.list_blobs(prefix="lunar_landar/models/PPO_0/")
# for blob in blobs:
#     # Get the blob's name
#     filename = blob.name.split("/")[-1]
#     # Download the blob to a local file in a new directory models_from_bucket
#     if not os.path.exists("models_from_bucket"):
#         os.makedirs("models_from_bucket")
#     blob.download_to_filename(f"models_from_bucket/{filename}")
#     print(f"Downloaded {filename} from the bucket")

# Test the newest model from the bucket
# latest_model = getLatestModel("models_from_bucket")
# if latest_model is not None:
#     env = gym.make('LunarLander-v2')
#     custom_objects = {"lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0}
#     model = PPO.load(f"models_from_bucket/{latest_model}", custom_objects=custom_objects, verbose=1)
#     model.set_env(env)
#     print(f"Testing model at timestep {model.num_timesteps}")
#     obs = env.reset()
#     for i in range(1000):
#         action, _states = model.predict(obs, deterministic=True)
#         obs, rewards, done, info = env.step(action)
#         env.render()
#         if done:
#             break
#     env.close()
