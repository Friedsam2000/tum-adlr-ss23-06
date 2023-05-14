import os
import gym
from stable_baselines3 import PPO
from google.cloud import storage
import torch
from stable_baselines3.common.env_util import make_vec_env

# Set up the GPU or use the CPU #######
print("GPU is available: ")
print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set up the bucket (google cloud storage) #######
# Define the bucket name
bucket_name = 'adlr_bucket'
# Initialize a storage client
storage_client = storage.Client()
# Get the bucket object
bucket = storage_client.get_bucket(bucket_name)

# Set up the number of parallel environments #######
num_cpu = 8
env = make_vec_env('LunarLander-v2', n_envs=num_cpu)

# Setting local directories
models_dir = "models/PPO_0"
logdir = "logs"


def getLatestModel(dir=models_dir):
    if os.path.exists(dir) and len(os.listdir(dir)) > 0:
        models = os.listdir(dir)
        # integer sort
        models.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        return models[-1]
    else:
        return None


####### Only comment in one of the following sections #######

####### Start training or continue training at the last local model #######
TIMESTEPS = 10000
MAX_TIMESTEPS = 2000000
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)
latest_model = getLatestModel()
if latest_model is not None:
    model = PPO.load(f"{models_dir}/{latest_model}", env=env, verbose=1, tensorboard_log=logdir)
    env.reset()
    print(f"Continue training at timestep {model.num_timesteps}")
else:
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir, device=device, n_steps=2048)
    print("Start training from scratch")
while model.num_timesteps < MAX_TIMESTEPS:
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model_name = f"{models_dir}/{model.num_timesteps}"
    model.save(model_name)

    # upload the n ew model to the bucket
    blob = bucket.blob(f"lunar_landar/{model_name}.zip")
    blob.upload_from_filename(f"{model_name}.zip")

    # get the latest log file
    logs = os.listdir(f"{logdir}/PPO_0")
    logs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    latest_log = logs[-1]
    # upload the new log file to the bucket
    blob = bucket.blob(f"lunar_landar/logs/PPO_0/{latest_log}")
    blob.upload_from_filename(f"{logdir}/PPO_0/{latest_log}")

####### Download the newest model from the bucket #######
# get all filenames in the bucket
# blobs = bucket.list_blobs(prefix="lunar_landar/models/PPO_0/")
# filenames = [blob.name for blob in blobs]
# # sort the filenames by the number of timesteps
# filenames.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
# blob = bucket.blob(filenames[-1])
# if not os.path.exists("models_from_bucket"):
#     os.makedirs("models_from_bucket")
# blob.download_to_filename(f"models_from_bucket/{filenames[-1].split('/')[-1]}")

####### Test the newest model from a folder #######
# folder = "great_model_from_c2_with_400k_timesteps"
# latest_model = getLatestModel(folder)
# if latest_model is not None:
#     env = gym.make('LunarLander-v2')
#     custom_objects = {"lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0}
#     model = PPO.load(f"{folder}/{latest_model}", custom_objects=custom_objects, verbose=1)
#     model.set_env(env)
#     print(f"Testing model at timestep {model.num_timesteps}")
#     for j in range(10):
#         sum_reward = 0
#         obs = env.reset()
#         for i in range(1000):
#             action, _states = model.predict(obs, deterministic=True)
#             obs, rewards, done, info = env.step(action)
#             sum_reward += rewards
#             env.render()
#             if done:
#                 print(f"Test {j} reward: {sum_reward}")
#                 break
#         env.close()
