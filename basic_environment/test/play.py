import sys

sys.path.insert(0, 'environments')
from environments.GridEnvironment import GridEnvironment
import cv2

env = GridEnvironment()
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

    # display the reward
    print(f"Reward = {reward}")

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
