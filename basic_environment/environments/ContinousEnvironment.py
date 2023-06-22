from collections import deque
import math
import numpy as np
import cv2
import gym
from gym import spaces

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, agent_size=0.2, goal_size=0.8, grid_size=(16,16), img_size=(48, 48), render_size=(960, 960), num_last_agent_pos=2):

        #Super init
        super(CustomEnv, self).__init__()

        # assert that the grid size is smaller than the image size
        assert grid_size[0] <= img_size[0] and grid_size[1] <= img_size[
            1], "The grid size must be smaller than the image size or important information will be lost."

        # assert that render size and image size are multiples of the grid size
        assert img_size[0] % grid_size[0] == 0 and img_size[1] % grid_size[1] == 0, \
            "The image size must be a multiple of the grid size or important information will be lost."
        assert render_size[0] % grid_size[0] == 0 and render_size[1] % grid_size[1] == 0, \
            "The render size must be a multiple of the grid size or important information will be lost."

        self.num_last_agent_pos = num_last_agent_pos
        self.grid_size = grid_size
        self.img_size = img_size
        self.render_size = render_size
        self.agent_size = agent_size
        self.goal_size = goal_size
        self.scaling = ( render_size[0]/ grid_size[0], render_size[1]/ grid_size[1])

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,2), dtype=np.single)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=grid_size[0], shape=(6,1), dtype=np.single)

    def reset(self):
        self.done = False

        # set the agent position to top left corner
        self.agent_position = np.array([0 + self.grid_size[0]//5, 0 + self.grid_size[1]//5])

        # set the goal position to bottom right corner
        self.goal_position =np.array( [self.grid_size[0] - self.grid_size[0]//5, self.grid_size[1] - self.grid_size[1]//5])

        # define last distance to goal
        self.initial_dist = np.linalg.norm(self.agent_position - self.goal_position)
        self.old_dist = self.initial_dist

        # define step counter and timeout as 6 times the time it takes to travel from start to goal
        self.steps = 0
        self.timeout = 6 * round(self.initial_dist)

        # define observation
        observation = self.get_observation()

        return observation

    def step(self, action):
        self.reward = 0
        self.steps += 1

        # apply action
        if action.shape[0] == 1:
            self.agent_position[0] = self.agent_position[0] + action[0,0]
            self.agent_position[1] = self.agent_position[1] + action[0,1]
        else:
            self.agent_position[0] = self.agent_position[0] + action[0]
            self.agent_position[1] = self.agent_position[1] + action[1]

        # observation
        observation = self.get_observation()
        
        # compute distance
        new_dist = np.linalg.norm(np.array(self.agent_position) - np.array(self.goal_position))

        # check if the agent is at the goal position
        if new_dist < self.goal_size:
            self.reward = 1

            self.done = True
            # return info that goal was reached
            return observation, self.reward, self.done, {"goal": True, "obstacle": False}

        if (self.agent_position[0] == 0.0) or (self.agent_position[1] == 0.0):
            self.reward = -1
            self.done = True
            return observation, self.reward, self.done, {"goal": False, "obstacle": True}

        # check if timeout
        if self.steps >= self.timeout:
            self.reward = -1
            self.done = True
            return observation, self.reward, self.done, {"goal": False, "obstacle": False}

        # if the agent is moving towards the goal, give a positive reward, if not, give a negative reward
        if new_dist < self.old_dist:
            self.reward = 0.025
        elif new_dist == self.old_dist: #wall hit
            self.reward = -0.05
        else:
            self.reward = -0.025

        # set the new distance to the old distance
        self.old_dist = new_dist

        return observation, self.reward, self.done, {}

    def render(self, mode='human'):
        # Render the environment to the screen
        #img = cv2.resize(self.getImg(), self.render_size, interpolation=cv2.INTER_NEAREST)
        img = self.getImg()
        cv2.imshow('image', img)
        cv2.waitKey(200)

    def close(self):
        cv2.destroyAllWindows()

    def get_observation(self):
        observation = np.zeros((6,1), dtype=np.single)
        observation[0] = self.agent_position[0]
        observation[1] = self.agent_position[1]
        observation[2] = self.agent_size
        observation[3] = self.goal_position[0]
        observation[4] = self.goal_position[1]
        observation[5] = self.goal_size
        return observation

    def getImg(self):
        agent_color = (255, 255, 255)
        goal_color = (0, 255, 0)
        obstacle_color = (0, 0, 255)
        old_agent_color = (255, 192, 203)

        img = np.zeros((self.grid_size[0], self.grid_size[1], 3), dtype=np.uint8)

        # scale the grid to img_size
        img = cv2.resize(img, self.render_size, interpolation=cv2.INTER_NEAREST)

        # plot agent
        x_agent = round(self.agent_position[0] * self.scaling[0])
        y_agent = round(self.agent_position[1] * self.scaling[1])
        radius = math.ceil(self.agent_size * self.scaling[0])
        img = cv2.circle(img, (x_agent, y_agent), radius, agent_color, -1)

        # plot goal
        x_goal = round(self.goal_position[0] * self.scaling[0])
        y_goal = round(self.goal_position[1] * self.scaling[1])
        radius_goal = math.ceil(self.goal_size * self.scaling[0])
        img = cv2.circle(img, (x_goal, y_goal), radius_goal, goal_color, -1)


        return img

