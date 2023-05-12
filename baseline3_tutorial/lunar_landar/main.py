import os
import gym
from stable_baselines3 import PPO
from google.cloud import storage

# Define your bucket name
bucket_name = 'adlr_bucket'

# Initialize a storage client
storage_client = storage.Client()

# Get the bucket object
bucket = storage_client.get_bucket(bucket_name)

env = gym.make('LunarLander-v2')

models_dir = "models/PPO_0"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


def getLatestModel():
    if os.path.exists(models_dir) and len(os.listdir(models_dir)) > 0:
        models = os.listdir(models_dir)
        # integer sort
        models.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        return models[-1]
    else:
        return None


# Start training or continue training at the last model
TIMESTEPS = 10000
MAX_TIMESTEPS = 100000
latest_model = getLatestModel()
if latest_model is not None:
    model = PPO.load(f"{models_dir}/{latest_model}", env=env, verbose=1, tensorboard_log=logdir)
    print(f"Continue training at timestep {model.num_timesteps}")
else:
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
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

# Test the latest model
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
