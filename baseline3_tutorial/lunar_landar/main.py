import gym
from stable_baselines3 import PPO
import os

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
        #integer sort
        models.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        return models[-1]
    else:
        return None

# Start training or continue training at the last model
TIMESTEPS = 10000
MAX_TIMESTEPS = 500000
latest_model = getLatestModel()
if latest_model is not None:
    model = PPO.load(f"{models_dir}/{latest_model}", env=env, verbose=1, tensorboard_log=logdir)
    print(f"Continue training at timestep {model.num_timesteps}")
else:
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
    print("Start training from scratch")
while model.num_timesteps < MAX_TIMESTEPS:
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{models_dir}/{model.num_timesteps}")


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