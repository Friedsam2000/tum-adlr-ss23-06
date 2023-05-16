import numpy as np
import cv2
import gym
from gym import spaces


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size):
        self.grid_size = grid_size
        super(CustomEnv, self).__init__()
        # Define action and observation space
        self.action_space = spaces.Discrete(4)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=(36, 36, 3), dtype=np.uint8)

    def reset(self):
        self.done = False
        self.reward = 0

        # set the block position to a random position
        self.agent_position = [np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])]
        self.goal_position = [np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])]

        # old agent positions
        self.old_agent_position = []

        # define observation
        observation = self.getImg()

        return np.array(observation, dtype=np.uint8)

    def step(self, action):

        # add the current agent position to the old agent positions before making a move
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
            # reward the agent for reaching the goal
            # self.reward = 100
            self.done = True
        else:
            distance = np.sqrt((self.agent_position[0] - self.goal_position[0]) ** 2 + (
                    self.agent_position[1] - self.goal_position[1]) ** 2)

            self.reward = - distance

        # define observation
        observation = self.getImg()

        return np.array(observation, dtype=np.uint8), self.reward, self.done, {}

    def render(self, mode='human'):

        cv2.imshow('Grid', self.getImg())
        cv2.waitKey(1)  # waits for 1 millisecond

    def close(self):
        cv2.destroyAllWindows()

    def getImg(self):
        block_color = (255, 255, 255)  # white color in RGB
        goal_color = (0, 255, 0)  # green color in RGB
        old_position_color = (100, 100, 100)  # gray color in RGB

        img = np.zeros((self.grid_size[0], self.grid_size[1], 3), dtype=np.uint8)

        # # draw the old agent positions in gray
        # for pos in self.old_agent_position:
        #     img[pos[1]:(pos[1] + 1), pos[0]:(pos[0] + 1)] = old_position_color

        # draw the active block (agent)
        img[self.agent_position[1]:(self.agent_position[1] + 1),
        self.agent_position[0] :(self.agent_position[0] + 1)] = block_color

        # draw the goal block
        img[self.goal_position[1]:(self.goal_position[1] + 1),
        self.goal_position[0]:(self.goal_position[0] + 1)] = goal_color

        # scale the grid to 64x64 pixels
        img = cv2.resize(img, (36, 36), interpolation=cv2.INTER_NEAREST)



        return img
