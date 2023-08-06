import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3 import SAC
#from environments.ContinousEnvironment import CustomEnv as ConEnv
from environments.ContinousEnvironment_2_Order import CustomEnv_2order_dyn as ConEnv
import os
import torch
from google.cloud import storage



def make_env(grid_size, rank):
    def _init():
        env = ConEnv(grid_size=grid_size, nr_obstacles=0, nr_goal_pos=1)
        return env

    return _init


if __name__ == "__main__":

    SAC_Iteration = "Test_1"
    SAC_Policy = "Test_1"
    print(SAC_Iteration)
    # Set up the GPU or use the CPU
    print("GPU is available: ")
    print(torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set up the bucket (google cloud storage)
    # Define the bucket name
    bucket_name = 'adlr_bucket'
    # Initialize a storage client
    storage_client = storage.Client()
    # Get the bucket object
    bucket = storage_client.get_bucket(bucket_name)

    num_cpu = 16  # Number of processes to use
    grid_size = (16, 16)

    # Create the vectorized environment
    env = SubprocVecEnv([make_env(grid_size, i) for i in range(num_cpu)])
    # add a monitor wrapper
    #env = VecMonitor(env)

    #print(env.get_attr("agent_damping_matrix"))
    #task_description = "agent_damping_matrix"
    agent_damping_matrix = np.zeros((2,2), dtype=np.single)
    agent_damping_matrix[0,0] = 0.5
    agent_damping_matrix[1,1] = 0.5
    #env.set_attr(task_description, agent_damping_matrix)
    #print(env.get_attr("agent_damping_matrix"))

    tasks = [{'agent_damping_matrix': agent_damping_matrix},{'agent_damping_matrix': agent_damping_matrix},{'agent_damping_matrix': agent_damping_matrix},{'agent_damping_matrix': agent_damping_matrix}, {'agent_damping_matrix': agent_damping_matrix}]

    #print(type(model.replay_buffer.buffers))
    #print(model.replay_buffer.buffers[0].observations[4900])
    #params = list(tasks[0].items())
    #for j in range(len(params)):
    #    print(params[j][0])
    #    print(params[j][1])

    ###env = ConEnv(grid_size=grid_size)
    #
    # Create logs if not existing
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Create models if not existing
    if not os.path.exists("models"):
        os.makedirs("models")

    # Check how many folders are in logs
    logs_folders = os.listdir("logs")

    # Initialize SAC agent with CNN Policy
    n_steps = 256
    #model = SAC("MlpPolicy", env, learning_rate=0.0003,verbose=1, buffer_size=1000000, optimize_memory_usage=False ,tensorboard_log="logs", device=device, batch_size=1024, gamma=0.999)

    model = SAC("MlpPolicy", env, learning_rate=0.0003, verbose=1, buffer_size=1000000, optimize_memory_usage=False,
                tensorboard_log="logs", device=device, batch_size=1024, gamma=0.999, use_pearl=True, nr_tasks=5, tasks=tasks)


    # create the folder for the model
    if not os.path.exists(f"models/SAC_{SAC_Iteration}"):
        os.makedirs(f"models/SAC_{SAC_Iteration}")

    best_reward = -np.inf
    log_save_counter = 0

    # Train agent
    TIMESTEPS_PER_SAVE = n_steps*num_cpu*20
    MAX_TIMESTEPS = 10000000
    while model.num_timesteps < MAX_TIMESTEPS:
        model.learn(total_timesteps=TIMESTEPS_PER_SAVE, reset_num_timesteps=False, tb_log_name=f"SAC_{SAC_Policy}")

        # get the mean reward of the last 100 episodes
        reward_mean = np.mean([ep['r'] for ep in list(model.ep_info_buffer)[-100:]])

        # if the reward mean is better than the best reward, save the model
        if reward_mean > best_reward:
            best_reward = reward_mean
            print(f"Saving model with new best reward mean {reward_mean}")
            model.save(f"models/SAC_{SAC_Iteration}/{model.num_timesteps}")

            # upload the model to the bucket
            blob = bucket.blob(f"data_Matthias/models/SAC_{SAC_Iteration}/{model.num_timesteps}.zip")
            blob.upload_from_filename(f"models/SAC_{SAC_Iteration}/{model.num_timesteps}.zip")
            print(f"Uploaded model {model.num_timesteps}.zip to bucket")
        if log_save_counter%2 == 0:
            # get the latest log file
            logs = os.listdir(f"logs/SAC_{SAC_Policy}_0")
            logs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            latest_log = logs[-1]
            # upload the new log file to the bucket
            blob = bucket.blob(f"data_Matthias/logs/SAC_{SAC_Iteration}/{latest_log}")
            blob.upload_from_filename(f"logs/SAC_{SAC_Policy}_0/{latest_log}")

        log_save_counter = log_save_counter + 1
