import sys
sys.path.insert(0, 'environments')
from environments.ContinousEnvironment_2_Order import CustomEnv_2order_dyn as ConEnv
import cv2

env = ConEnv()
# Display the image in a window
observation = env.reset()
# print(observation)
env.render()
i = 0
# Event looop
while i < 10000:
    key = cv2.waitKey(0)
    if key == 82:  # up
        action = [0,0.1]
    elif key == 84:  # down
        action = [0,-0.1]
    elif key == 81:  # left
        i = 10001
    elif key == 83:  # right
        action = [0.1,0]
    else:
        continue

    # Apply the action to the environment
    observation, reward, done, info = env.step(action)
    # print(observation)

    # display the reward
    print(f"Reward = {reward}")
    print(observation)

    # Check if the episode is done
    if done:
        print('Episode finished')
        observation = env.reset()

    env.render()
    i += 1

cv2.destroyAllWindows()