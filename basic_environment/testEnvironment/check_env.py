from environments.GridEnvironment import GridEnvironment
from stable_baselines3.common.env_checker import check_env


env = GridEnvironment([16,16])
# It will check your custom environment and output additional warnings if needed
check_env(env, warn=True, skip_render_check=True)
