import sys
sys.path.insert(0, 'environments')
from environments.GridEnvironment import CustomEnv as GridEnvironment
import cv2

env = GridEnvironment(grid_size=(16, 16))
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
    observation, reward, done, info = env.step(action)
    # print(observation)

    # display the reward
    print(f"Reward = {reward}")

    # Check if the episode is done
    if done:
        print('Episode finished')
        observation = env.reset()

    env.render()