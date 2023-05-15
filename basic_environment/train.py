import os
import time

import gym
from stable_baselines3 import PPO
from GridEnvironment import CustomEnv as GridEnvironment


# Setting local directories
models_dir = "models/PPO_0"
logdir = "logs"

# create the environment
env = GridEnvironment(grid_size=(20, 20))

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
MAX_TIMESTEPS = 1000000
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
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=logdir)
    print("Start training from scratch")
while model.num_timesteps < MAX_TIMESTEPS:
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model_name = f"{models_dir}/{model.num_timesteps}"
    model.save(model_name)

###### Test the newest model from a folder #######
# folder = "models/PPO_0"
# latest_model = getLatestModel(folder)
# if latest_model is not None:
#     model = PPO.load(f"{folder}/{latest_model}", verbose=1)
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
#             #wait 100ms
#             time.sleep(0.1)
#             if done:
#                 print(f"Test {j} reward: {sum_reward}")
#                 break
#         env.close()"models/PPO_0"
# latest_model = getLatestModel(folder)
# if latest_model is not None:
#     model = PPO.load(f"{folder}/{latest_model}", verbose=1)
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
#             #wait 100ms
#             time.sleep(0.1)
#             if done:
#                 print(f"Test {j} reward: {sum_reward}")
#                 break
#         env.close()