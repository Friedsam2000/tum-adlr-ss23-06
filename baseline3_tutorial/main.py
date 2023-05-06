import gym
from stable_baselines3 import PPO
import os


models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make('LunarLander-v2')
env.reset()

# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

#Training
# TIMESTEPS = 10000
# iters = 0
# for i in range(30):
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
#     model.save(f"{models_dir}/{TIMESTEPS*i}")


#Continuing training
# timesteps_to_continue = 320000
# model = PPO.load(f"{models_dir}/{timesteps_to_continue}", env=env)
# TIMESTEPS = 10000
# for i in range(10):
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
#     model.save(f"{models_dir}/{TIMESTEPS*i+timesteps_to_continue}")


# Testing
test_timesteps = 350000
model = PPO.load(f"{models_dir}/{test_timesteps}")
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        break
env.close()

