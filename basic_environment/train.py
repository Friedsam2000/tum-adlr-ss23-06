from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from GridEnvironment import CustomEnv

def make_env(grid_size, rank):
    def _init():
        env = CustomEnv(grid_size=grid_size)
        return env
    return _init

def main():
    num_cpu = 4  # Number of processes to use
    grid_size = (8, 8)

    # Create the vectorized environment
    env = SubprocVecEnv([make_env(grid_size, i) for i in range(num_cpu)])
    env = VecMonitor(env)

    # Initialize PPO agent with CNN policy
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="logs")

    # Train agent
    model.learn(total_timesteps=200000)

    # Save the agent
    model.save("ppo_customenv")

if __name__ == "__main__":
    main()
