import gym
import torch as th
from stable_baselines3 import PPO

# create the environment
env = gym.make('CartPole-v1')

# create the agent
model = PPO('MlpPolicy', env, verbose=1)

# train the agent
model.learn(total_timesteps=10000)

# test the agent
obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
