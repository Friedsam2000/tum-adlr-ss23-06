from stable_baselines3.common.env_checker import check_env
from environments.FeatureExtractedEnv import FeatureExtractedEnv

env = FeatureExtractedEnv()
check_env(env, warn=True, skip_render_check=True)