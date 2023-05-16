from stable_baselines3 import PPO
from GridEnvironment import CustomEnv
import os


def getLatestModel(dir="models"):
    if os.path.exists(dir) and len(os.listdir(dir)) > 0:
        models = os.listdir(dir)
        # integer sort
        models.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        return models[-1]
    else:
        return None


def main():

    # Create the environment
    env = CustomEnv(grid_size=(8,8))

    # get the latest model
    models_dir = "models"
    latest_model = getLatestModel(dir=models_dir)
    print(f"Loading model {latest_model}")

    # Load the model
    model = PPO.load(f"{models_dir}/{latest_model}", env=env, verbose=1)


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
