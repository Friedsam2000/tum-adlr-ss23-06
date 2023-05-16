from stable_baselines3.common.env_checker import check_env
from GridEnvironment import CustomEnv as GridEnvironment


env = GridEnvironment([50,50])
# It will check your custom environment and output additional warnings if needed
check_env(env)