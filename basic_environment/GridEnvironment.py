from collections import deque

import numpy as np
import cv2
import gym
from gym import spaces


REMEMBER_NUM_OLD_AGENT_POSITIONS = 30


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size, img_size=(36, 36), render_size=(360, 360)):
        self.grid_size = grid_size
        self.img_size = img_size
        self.render_size = render_size
        super(CustomEnv, self).__init__()
        # Define action and observation space
        self.action_space = spaces.Discrete(4)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=(img_size[0], img_size[1], 3), dtype=np.uint8)

    def reset(self):
        self.done = False

        # set the block position to a random position
        self.agent_position = [np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])]
        self.goal_position = [np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])]

        # old agent positions
        self.old_agent_position = deque(maxlen=REMEMBER_NUM_OLD_AGENT_POSITIONS)
        for i in range(REMEMBER_NUM_OLD_AGENT_POSITIONS):
            self.old_agent_position.append([-1, -1]) # -1 means no position

        # define observation
        observation = self.getImg()

        return np.array(observation, dtype=np.uint8)

    def step(self, action):

        self.reward = 0

        # save old agent position
        self.old_agent_position.append(self.agent_position.copy())

        if action == 0:  # up
            self.agent_position[1] = max(0, self.agent_position[1] - 1)
        elif action == 1:  # down
            self.agent_position[1] = min(self.grid_size[1] - 1, self.agent_position[1] + 1)
        elif action == 2:  # left
            self.agent_position[0] = max(0, self.agent_position[0] - 1)
        elif action == 3:  # right
            self.agent_position[0] = min(self.grid_size[0] - 1, self.agent_position[0] + 1)

        if self.agent_position == self.goal_position:
            self.reward += 20
            self.done = True

        # define reward as the decrease in distance to the goal
        old_dist = abs(self.old_agent_position[-1][0] - self.goal_position[0]) + abs(
            self.old_agent_position[-1][1] - self.goal_position[1])
        new_dist = abs(self.agent_position[0] - self.goal_position[0]) + abs(
            self.agent_position[1] - self.goal_position[1])
        self.reward += old_dist - new_dist

        # if the agent is not moving at all, give a negative reward
        if old_dist == new_dist:
            self.reward += -0.4

        # if the agent is moving back to the old position, give a negative reward
        if self.agent_position in self.old_agent_position:
            self.reward += -0.2

        # any move is a negative reward
        self.reward += -0.1

        # define observation
        observation = self.getImg()

        return np.array(observation, dtype=np.uint8), self.reward, self.done, {}

    def render(self, mode='human'):
        # Render the environment to the screen
        img = cv2.resize(self.getImg(), self.render_size, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('image', img)
        cv2.waitKey(100)


    def close(self):
        cv2.destroyAllWindows()

    def getImg(self):
        block_color = (255, 255, 255)
        goal_color = (0, 255, 0)
        old_position_color = (100, 100, 100)

        img = np.zeros((self.grid_size[0], self.grid_size[1], 3), dtype=np.uint8)

        # draw the old agent positions in gray
        for old_position in self.old_agent_position:
            img[old_position[1]:(old_position[1] + 1), old_position[0]:(old_position[0] + 1)] = old_position_color

        # draw the agent position in white
        img[self.agent_position[1]:(self.agent_position[1] + 1), self.agent_position[0]:(self.agent_position[0] + 1)] = block_color

        # draw the goal position in green
        img[self.goal_position[1]:(self.goal_position[1] + 1), self.goal_position[0]:(self.goal_position[0] + 1)] = goal_color

        # scale the grid to img_size
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_NEAREST)

        return img