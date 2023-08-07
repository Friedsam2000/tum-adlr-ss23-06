from stable_baselines3.common.env_checker import check_env
from environments.FeatureExtractedEnv import FeatureExtractedEnv, GridEnvironment

env = FeatureExtractedEnv(GridEnvironment(num_last_agent_pos=0,num_obstacles=0, num_frames_to_stack=2))
check_env(env, warn=True, skip_render_check=True)