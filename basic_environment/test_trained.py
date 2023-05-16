from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from GridEnvironment import CustomEnv

def main():

    # Create the environment
    env = CustomEnv(grid_size=(8,8))

    # Load the trained model
    model = PPO.load("ppo_customenv", env=env)

    # Test the trained model
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print('reward',reward)
        env.render()
        if done:
          obs = env.reset()

    env.close()

if __name__ == "__main__":
    main()
