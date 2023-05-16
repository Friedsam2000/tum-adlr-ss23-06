from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from GridEnvironment import CustomEnv
import os



def make_env(grid_size, rank):
    def _init():
        env = CustomEnv(grid_size=grid_size)
        # add a TimeLimit wrapper
        env = TimeLimit(env, max_episode_steps=100)
        return env
    return _init

def main():
    num_cpu = 4  # Number of processes to use
    grid_size = (8, 8)

    # Create the vectorized environment
    env = SubprocVecEnv([make_env(grid_size, i) for i in range(num_cpu)])
    # add a monitor wrapper
    env = VecMonitor(env)

    # create logs and models directories if they don't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")

    if not os.path.exists("models"):
        os.makedirs("models")

    # Initialize PPO agent with CNN policy
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="logs")

    # Train agent

    timesteps_per_save = 50000
    max_timesteps = 300000
    while model.num_timesteps < max_timesteps:
        model.learn(total_timesteps=timesteps_per_save, reset_num_timesteps=False)
        model.save(f"models/{model.num_timesteps}")


if __name__ == "__main__":
    main()
