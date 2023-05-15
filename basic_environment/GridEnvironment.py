from collections import deque

import numpy as np
import cv2
import gym
from gym import spaces

REMEMBER_NUM_PREV_ACTIONS = 0
DISPLAY_NUM_PREV_BLOCK_POSITIONS = 30

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size):
        self.grid_size = grid_size
        super(CustomEnv, self).__init__()
        # Define action and observation space
        self.action_space = spaces.Discrete(4)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=-1, high=max(*grid_size, 4), shape=(6 + REMEMBER_NUM_PREV_ACTIONS,),
                                            dtype=np.float32)
    def reset(self):
        self.done = False
        self.reward = 0
        # set the block position to a random position
        self.block_position = [np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])]
        self.goal_position = [np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])]

        self.old_block_positions = deque(maxlen=DISPLAY_NUM_PREV_BLOCK_POSITIONS)
        self.prev_actions = deque(maxlen=REMEMBER_NUM_PREV_ACTIONS)

        for i in range(DISPLAY_NUM_PREV_BLOCK_POSITIONS):
            self.old_block_positions.append((-1, -1))
        for i in range(REMEMBER_NUM_PREV_ACTIONS):
            self.prev_actions.append(-1)  # -1 represents no action

        # define observation
        observation = [self.block_position[1], self.block_position[0], self.goal_position[1],
                       self.goal_position[0], self.grid_size[1], self.grid_size[0]] + list(self.prev_actions)

        return np.array(observation, dtype=np.float32)

    def step(self, action):
        # save the old block positions
        self.old_block_positions.append(self.block_position.copy())
        self.prev_actions.append(action)

        if action == 0:  # up
            self.block_position[1] = max(0, self.block_position[1] - 1)
        elif action == 1:  # down
            self.block_position[1] = min(self.grid_size[1] - 1, self.block_position[1] + 1)
        elif action == 2:  # left
            self.block_position[0] = max(0, self.block_position[0] - 1)
        elif action == 3:  # right
            self.block_position[0] = min(self.grid_size[0] - 1, self.block_position[0] + 1)

        if self.block_position == self.goal_position:
            # reward the agent for reaching the goal
            self.reward = 10
            self.done = True
        else:
            # penalize the agent for moving
            self.reward = -0.1

        # define observation
        observation = [self.block_position[1], self.block_position[0], self.goal_position[1],
                       self.goal_position[0], self.grid_size[1], self.grid_size[0]] + list(self.prev_actions)

        return np.array(observation, dtype=np.float32), self.reward, self.done, {}

    def render(self, mode='human'):
        #... keep the rest as is


        block_size = 10  # size of a block in pixels
        block_color = (255, 255, 255)  # white color in RGB
        grid_color = (0, 0, 0)  # black color in RGB
        goal_color = (0, 255, 0)  # green color in RGB
        old_block_color = (128, 128, 128) # gray color in RGB

        img = np.zeros((self.grid_size[0] * block_size, self.grid_size[1] * block_size, 3), dtype=np.uint8)
        # draw old block positions
        for old_block_position in self.old_block_positions:
            if old_block_position != (-1, -1):
                img[old_block_position[1] * block_size:(old_block_position[1] + 1) * block_size,
                old_block_position[0] * block_size:(old_block_position[0] + 1) * block_size] = old_block_color

        # draw the active block
        img[self.block_position[1] * block_size:(self.block_position[1] + 1) * block_size,
        self.block_position[0] * block_size:(self.block_position[0] + 1) * block_size] = block_color
        # draw the goal block
        img[self.goal_position[1] * block_size:(self.goal_position[1] + 1) * block_size,
        self.goal_position[0] * block_size:(self.goal_position[0] + 1) * block_size] = goal_color

        cv2.imshow('Grid', img)
        cv2.waitKey(1)  # waits for 1 millisecond

    def close(self):
        cv2.destroyAllWindows()


