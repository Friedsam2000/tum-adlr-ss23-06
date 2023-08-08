import sys
sys.path.insert(0, 'environments')
from environments.FeatureExtractedEnv import FeatureExtractedEnv, GridEnvironment
import cv2
import numpy as np
import torch

env = FeatureExtractedEnv(GridEnvironment(num_last_agent_pos=0, num_obstacles=6, num_frames_to_stack=1, size_grid_frame_info=11))

observation, _ = env.reset()  # Modified to account for the reset method returning a tuple of (obs, info)
env.render()

while True:
    key = cv2.waitKey(0)
    if key == 82:  # up
        action = 0
    elif key == 84:  # down
        action = 1
    elif key == 81:  # left
        action = 2
    elif key == 83:  # right
        action = 3
    else:
        continue

    # Inside the while loop:

    obs, reward, terminated, truncated, info = env.step(action)
    frame_info = env.get_current_frame_info()

    # Extracting the obstacle grid and position from the observation
    obs_grid = obs[::2, :, :]
    obs_pos = obs[1::2, :, :]


    # Using the first frame since num_frames_to_stack=1
    predicted_agent_pos_frame = obs_pos[0]*23.0
    predicted_grid_frame = obs_grid[0]

   # Both Positions are in a 2x2 grid at position (2, 2) to (3, 3)
    predicted_agent_pos_frame = predicted_agent_pos_frame[2:4, 2:4].reshape(4)

    predicted_agent_pos = predicted_agent_pos_frame[:2]
    predicted_goal_pos = predicted_agent_pos_frame[2:]



    print("predicted agent position: ", np.round(np.array(predicted_agent_pos)))
    print("true agent position     : ", frame_info['agent_position'])


    print("predicted goal position : ", np.round(np.array(predicted_goal_pos)))
    print("true goal position      : ", frame_info['goal_position'])

    # Convert the predicted grid to binary based on a threshold
    threshold = 0.5
    predicted_grid_binary = np.where(predicted_grid_frame > threshold, 1, 0)

    print("predicted grid: ")
    print(predicted_grid_binary)

    print("true grid: ")
    true_grid = frame_info['neighboring_cells_content']
    true_grid = np.array(true_grid)
    true_grid = true_grid.reshape((13, 13))
    print(true_grid)

    print("--------------------------------------------------")

    if terminated:
        if reward == 1:
            print("Goal reached")
        elif reward == -1:
            print("Obstacle hit")
        else:
            print("Error")

    if truncated:
        print("Timeout")

    if terminated or truncated:
        obs, _ = env.reset()  # Modified to account for the reset method returning a tuple of (obs, info)

    env.render()
