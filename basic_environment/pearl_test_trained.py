from stable_baselines3 import PPO
from stable_baselines3 import SAC
from environments.GridEnvironment import CustomEnv
from environments.GridEnvironment import CustomEnv_rc
#from environments.ContinousEnvironment import CustomEnv as ConEnv
from environments.ContinousEnvironment_2_Order import CustomEnv_2order_dyn as ConEnv
import os
import google.cloud.storage
import shutil
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
import torch as torch

## Set up the Bucket (google cloud storage)
## Define the bucket name
#bucket_name = 'adlr_bucket'
## Initialize a storage client
#storage_client = google.cloud.storage.Client()
## Get the bucket object
#bucket = storage_client.get_bucket(bucket_name)
#
## Get all model filenames from the bucket
#PPO_Iteration = "SAC_MLP_7.1"
#blobs = bucket.list_blobs(prefix=f"data_Matthias/models/{PPO_Iteration}")
#model_filenames = []
#for blob in blobs:
#    model_filenames.append(blob.name)
#
## Integer sort the model filenames
#model_filenames = sorted(model_filenames, key=lambda x: int(x.split("/")[-1].split(".")[0]))
#
## Empy or create the models_from_bucket directory using shutil
#if os.path.exists("models_from_bucket"):
#    shutil.rmtree("models_from_bucket")
#os.makedirs("models_from_bucket")
#
#
#
## Download the model with the highest number of steps in the models_from_bucket directory
#model_filename = model_filenames[-1]
#blob = bucket.blob(model_filename)
#blob.download_to_filename(f"models_from_bucket/" + model_filename.split("/")[-1])
#print(f"Downloaded {model_filename} from bucket {bucket_name} to models_from_bucket directory")

# Load the model
agent_damping_matrix = np.zeros((2,2), dtype=np.single)
agent_damping_matrix[0,0] = 0.5
agent_damping_matrix[1,1] = 0.5
custom_objects = {"lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0, "tasks": [{'agent_damping_matrix': agent_damping_matrix}, {'agent_damping_matrix': agent_damping_matrix}, {'agent_damping_matrix': agent_damping_matrix}]}
model = SAC.load("models/SAC_Test_18_5/10.243095397949219", custom_objects=custom_objects, verbose=1)
print("Loaded Model")
print(type(model.agent.context_encoder))
# Create the environment
#env = CustomEnv(grid_size=(16, 16))
#env = CustomEnv_rc(grid_size=(16, 16))
env = ConEnv(grid_size=(16, 16), nr_obstacles=0, nr_goal_pos=1, test=True)

buffer = ReplayBuffer(buffer_size=10000,
        observation_space=env.observation_space,
        action_space=env.action_space,
        handle_timeout_termination=False)
device = "cuda" if torch.cuda.is_available() else "cpu"
# Sample Buffer
obs = env.reset()
model.agent.clear_z()
episodes = 0
while buffer.pos <= model.history and not(buffer.full):
    last_obs = obs
    #first trajectories with prior
    model.agent.sample_z()
    action, _states = model.predict(np.concatenate((obs,model.agent.z[0].detach().cpu())), deterministic=True)
    obs, reward, done, info = env.step(action)
    buffer.add(last_obs, obs, action, reward, done, info)

    #env.render()
    if done:
        episodes += 1
        obs = env.reset()

print(episodes)
# Test the model
obs = env.reset()
goals_reached = 0
obstacles_hit = 0
episodes = 0
num_steps = 0
#num_step_per_episode = []
#steps = 0
while episodes < 1000:
    last_obs = obs
    context = buffer.sample_context(meta_batch_size=4,history=32)
    model.agent.infer_posterior(context.float().to(device))
    action, _states = model.predict(np.concatenate((obs,model.agent.z[0].detach().cpu())), deterministic=True)
    obs, reward, done, info = env.step(action)
    buffer.add(last_obs, obs, action, reward, done, info)
    num_steps += 1
    #steps += 1
    env.render()
    if done:
        episodes += 1
        if info["goal"]:
            goals_reached += 1
        if info["obstacle"]:
            obstacles_hit += 1
        #num_step_per_episode.append(steps)
        #steps = 0
        obs = env.reset()


print(f"Succes rate: {goals_reached / episodes}")
print(f"Obstacles hit: {obstacles_hit / episodes}")
print(f"Timeouts: {1 - (goals_reached + obstacles_hit) / episodes}")
print(f"Mean Steps: {num_steps / episodes}")
#print(f"Steps per Episode:")
#print(num_step_per_episode)

