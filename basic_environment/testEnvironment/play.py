import sys
sys.path.insert(0, 'environments')
from environments.FeatureExtractedEnv import FeatureExtractedEnv, GridEnvironment
import cv2
import numpy as np
import torch

env = FeatureExtractedEnv(GridEnvironment(num_last_agent_pos=0,num_obstacles=6, num_frames_to_stack=1))
# Display the image in a window
observation = env.reset()
# print(observation)
env.render()

# Event looop
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

    # Apply the action to the environment
    obs, reward, terminated, truncated, info = env.step(action)

    # Get the current frame info
    frame_info = env.get_current_frame_info()

    # Compare the true and predicted positions
    print("predicted agent position: ", np.round(obs[0:2]*23))
    print("true agent position     : ", frame_info['agent_position'])

    print("predicted goal position : ", np.round(obs[2:4]*23))
    print("true goal position      : ", frame_info['goal_position'])

    # Get the predicted grid
    predictions_grid = obs[4:]

    # convert the predicted grid to binary

    # Print the predicted grid
    print("predicted grid: ")
    print(predictions_grid.reshape(7, 7))

    # Print the true grid after reshape
    print("true grid: ")
    true_grid = frame_info['neighboring_cells_content']
    # list to numpy array
    true_grid = np.array(true_grid)
    # reshape
    true_grid = true_grid.reshape(7, 7)
    print(true_grid)


    # print a line
    print("--------------------------------------------------")


    # display the reward
    # print(f"Reward = {reward}")

    # display the observation
    # print(f"Observation = {obs}")

    # Check if the episode is done
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
        obs, info = env.reset()

    # Display the image in a window
    env.render()
