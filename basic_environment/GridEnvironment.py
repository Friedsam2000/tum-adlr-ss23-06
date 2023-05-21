from collections import deque
import numpy as np
import cv2
import gym
from gym import spaces


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size, img_size=(84, 84), render_size=(360, 360), num_last_agent_pos=2):

        # assert that the grid size is smaller than the image size
        assert grid_size[0] <= img_size[0] and grid_size[1] <= img_size[
            1], "The grid size must be smaller than the image size or important information will be lost."

        self.num_last_agent_pos = num_last_agent_pos
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

        # create a deque to store the last agent positions
        self.last_agent_positions = deque(maxlen=self.num_last_agent_pos)
        # create history of agent positions and set to -1
        for i in range(self.num_last_agent_pos):
            self.last_agent_positions.append([-1, -1])

        # set the block position to a random position
        self.agent_position = [np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])]
        self.goal_position = [np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])]

        # spawn random obstacles not on goal or agent position
        self.obstacle_positions = []
        for i in range(np.random.randint(0, 10)):
            obstacle_position = [np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])]
            while obstacle_position == self.agent_position or obstacle_position == self.goal_position:
                obstacle_position = [np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])]
            self.obstacle_positions.append(obstacle_position)

        # define last distance to goal
        self.old_dist = np.linalg.norm(np.array(self.agent_position) - np.array(self.goal_position))

        # define step counter and timeout as 4 times the manhattan distance between agent and goal
        self.steps = 0
        self.timeout = 6 * (abs(self.agent_position[0] - self.goal_position[0]) + abs(
            self.agent_position[1] - self.goal_position[1])) + 1

        # define observation
        observation = self.getImg()

        return np.array(observation, dtype=np.uint8)

    def step(self, action):

        self.reward = 0
        self.steps += 1

        # append the current agent position to the last agent positions
        self.last_agent_positions.append(self.agent_position.copy())

        if action == 0:  # up
            self.agent_position[1] = max(0, self.agent_position[1] - 1)
        elif action == 1:  # down
            self.agent_position[1] = min(self.grid_size[1] - 1, self.agent_position[1] + 1)
        elif action == 2:  # left
            self.agent_position[0] = max(0, self.agent_position[0] - 1)
        elif action == 3:  # right
            self.agent_position[0] = min(self.grid_size[0] - 1, self.agent_position[0] + 1)

        # check if the agent hit an obstacle
        if self.agent_position in self.obstacle_positions:
            self.reward = -1
            self.done = True
            return np.array(self.getImg(), dtype=np.uint8), self.reward, self.done, {"goal": False, "obstacle": True}

        # check if the agent is at the goal position
        if self.agent_position == self.goal_position:
            self.reward = 1

            self.done = True
            # return info that goal was reached
            return np.array(self.getImg(), dtype=np.uint8), self.reward, self.done, {"goal": True, "obstacle": False}



        # check if timeout
        if self.steps >= self.timeout:
            self.reward = -1
            self.done = True
            return np.array(self.getImg(), dtype=np.uint8), self.reward, self.done, {"goal": False, "obstacle": False}

        # define new distance to goal
        new_dist = np.linalg.norm(np.array(self.agent_position) - np.array(self.goal_position))

        # if the agent is moving towards the goal, give a positive reward, if not, give a negative reward
        if new_dist < self.old_dist:
            self.reward = 0.05
        elif new_dist == self.old_dist: #wall hit
            self.reward = -0.1
        else:
            self.reward = -0.05

        # punish the agent for revisiting old positions
        if self.agent_position in self.last_agent_positions:
            self.reward -= 0.05


        # set the new distance to the old distance
        self.old_dist = new_dist

        return np.array(self.getImg(), dtype=np.uint8), self.reward, self.done, {}

    def render(self, mode='human'):
        # Render the environment to the screen
        img = cv2.resize(self.getImg(), self.render_size, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('image', img)
        cv2.waitKey(200)

    def close(self):
        cv2.destroyAllWindows()

    def getImg(self):
        block_color = (255, 255, 255)
        goal_color = (0, 255, 0)
        obstacle_color = (0, 0, 255)
        old_agent_color = (255, 192, 203)


        img = np.zeros((self.grid_size[0], self.grid_size[1], 3), dtype=np.uint8)

        # draw the last agent positions in light pink if not -1
        for position in self.last_agent_positions:
            if position != [-1, -1]:
                img[position[1]:(position[1] + 1),
                position[0]:(position[0] + 1)] = old_agent_color

        # draw the agent position in white
        img[self.agent_position[1]:(self.agent_position[1] + 1),
        self.agent_position[0]:(self.agent_position[0] + 1)] = block_color

        # draw the goal position in green
        img[self.goal_position[1]:(self.goal_position[1] + 1),
        self.goal_position[0]:(self.goal_position[0] + 1)] = goal_color

        # draw the obstacle positions in red
        for obstacle_position in self.obstacle_positions:
            img[obstacle_position[1]:(obstacle_position[1] + 1),
            obstacle_position[0]:(obstacle_position[0] + 1)] = obstacle_color

        # scale the grid to img_size
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_NEAREST)

        return img
