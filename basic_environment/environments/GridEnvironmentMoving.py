from collections import deque
import numpy as np
import cv2
import gymnasium
from gymnasium import spaces
from utility.CheckGoalReachable import a_star_search
from .Obstacle import Obstacle

class CustomEnv(gymnasium.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    action_space = spaces.Discrete(4)

    def __init__(self, grid_size, img_size=(48, 48), render_size=(480, 480), num_last_agent_pos=0, num_frames_to_stack=2):
        super().__init__()
        self.num_frames_to_stack = num_frames_to_stack
        self.frame_stack = deque(maxlen=num_frames_to_stack)
        self._assert_sizes(grid_size, img_size, render_size)

        self.num_last_agent_pos = num_last_agent_pos
        self.grid_size = grid_size
        self.img_size = img_size
        self.render_size = render_size

        self.observation_space = spaces.Box(low=0, high=255, shape=(img_size[0], img_size[1], 3*num_frames_to_stack), dtype=np.uint8)

    def reset(self, seed=None):
        # Reset the environment and optionally set the random seed
        if seed is not None:
            np.random.seed(seed)

        self.done = False
        self._init_positions()
        self._spawn_obstacles()

        # define last distance to goal
        self.old_dist = np.linalg.norm(np.array(self.agent_position) - np.array(self.goal_position))

        # define step counter and timeout as 6 times the manhattan distance between agent and goal
        self.steps = 0
        self.timeout = 6 * (abs(self.agent_position[0] - self.goal_position[0]) + abs(
            self.agent_position[1] - self.goal_position[1])) + 1

        # Reset the frame stack with four identical frames
        for _ in range(self.num_frames_to_stack):
            stacked_frames = self.getImg()

        initial_observation = np.array(stacked_frames, dtype=np.uint8)
        info = {}  # Add any relevant info here

        return initial_observation, info

    def step(self, action):
        self.reward = 0
        self.steps += 1
        self._move_agent(action)
        self._evaluate_reward()
        self._check_goal()
        self._check_obstacle_collision()
        self._timeout_check()

        # Determine the value of terminated and truncated
        terminated = self.done
        truncated = False

        obs = np.array(self.getImg(), dtype=np.uint8)
        info = {"goal": self.done and self.reward == 1, "obstacle": self.done and self.reward == -1}

        return obs, self.reward, terminated, truncated, info

    def render(self, mode='human'):
        # Get the last frame from the deque
        img = self.getImg()

        # Use the newest 3 channels for displaying
        display_img = img[:, :, -3:]

        # Resize the image for better visualization
        display_img = cv2.resize(display_img, self.render_size, interpolation=cv2.INTER_NEAREST)

        # Display the image
        cv2.imshow('image', display_img)
        cv2.waitKey(100)

    def close(self):
        cv2.destroyAllWindows()

    def getImg(self):
        # Create a base image to represent the grid
        base_image = np.zeros((self.grid_size[0], self.grid_size[1], 3))


        # Draw the old agent positions
        for old_pos in self.last_agent_positions:
            if old_pos[0] != -1 and old_pos[1] != -1:  # only draw if it's not the initial [-1, -1]
                base_image[old_pos[0], old_pos[1]] = [255, 255, 0]  # Yellow for old agent positions

        # Draw the agent, goal and obstacles on the base image
        # assuming the agent, goal and obstacles are represented as different colors in RGB
        base_image[self.agent_position[0], self.agent_position[1]] = [255, 0, 0]  # Red for agent
        base_image[self.goal_position[0], self.goal_position[1]] = [0, 255, 0]  # Green for goal
        for obstacle in self.obstacles:
            base_image[obstacle.position[0], obstacle.position[1]] = [0, 0, 255]  # Blue for obstacles

        # Scale the image up for easier viewing (optional)
        scaled_image = cv2.resize(base_image, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_AREA)

        # Add the new frame to the stack
        self.frame_stack.append(scaled_image)

        # Concatenate the frames along the channel dimension
        stacked_frames = np.concatenate(self.frame_stack, axis=-1)

        # Return the stacked frames as a numpy array
        return stacked_frames

    def _assert_sizes(self, grid_size, img_size, render_size):
        # assert that the grid size is smaller than the image size
        assert grid_size[0] <= img_size[0] and grid_size[1] <= img_size[
            1], "The grid size must be smaller than the image size or important information will be lost."

        # assert that render size and image size are multiples of the grid size
        assert img_size[0] % grid_size[0] == 0 and img_size[1] % grid_size[1] == 0, \
            "The image size must be a multiple of the grid size or important information will be lost."
        assert render_size[0] % grid_size[0] == 0 and render_size[1] % grid_size[1] == 0, \
            "The render size must be a multiple of the grid size or important information will be lost."

    def _init_positions(self):
        # create a deque to store the last agent positions
        self.last_agent_positions = deque(maxlen=self.num_last_agent_pos)
        # create history of agent positions and set to -1
        for _ in range(self.num_last_agent_pos):
            self.last_agent_positions.append([-1, -1])

        # set the agent position to a random position
        self.agent_position = [np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])]
        # set the goal position to a random position
        self.goal_position = [np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])]

        # make sure that the goal position is not the same as the agent position
        while self.goal_position == self.agent_position:
            self.goal_position = [np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])]

    def _spawn_obstacles(self):
        # Spawn random obstacles not on goal or agent position
        goal_is_reachable = False
        while not goal_is_reachable:
            self.obstacles = []  # Clear the old obstacles
            for _ in range(10):
                obstacle_position = self._generate_obstacle_position()
                # Determine whether this obstacle will be moving, and in what direction
                is_moving = True
                direction = np.random.choice([0, 1, 2, 3])
                self.obstacles.append(Obstacle(obstacle_position, is_moving, direction))

            self.obstacle_positions = [obstacle.position for obstacle in self.obstacles]
            goal_is_reachable = self._check_goal_reachable(self.goal_position, self.agent_position, self.obstacle_positions)

    def _generate_obstacle_position(self):
        obstacle_position = [np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])]
        # Make sure obstacle is not on agent or goal position and at least 2 fields away from the agent
        while (obstacle_position == self.agent_position or
               obstacle_position == self.goal_position or
               abs(obstacle_position[0] - self.agent_position[0]) < 2 or
               abs(obstacle_position[1] - self.agent_position[1]) < 2):
            obstacle_position = [np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])]
        return obstacle_position

    def _move_agent(self, action):
        # append the current agent position to the last agent positions
        self.last_agent_positions.append(self.agent_position.copy())

        if action == 0:  # up
            self.agent_position[0] = max(0, self.agent_position[0] - 1)
        elif action == 1:  # down
            self.agent_position[0] = min(self.grid_size[0] - 1, self.agent_position[0] + 1)
        elif action == 2:  # left
            self.agent_position[1] = max(0, self.agent_position[1] - 1)
        elif action == 3:  # right
            self.agent_position[1] = min(self.grid_size[1] - 1, self.agent_position[1] + 1)

    def _check_obstacle_collision(self):
        # check if the agent hit an obstacle
        for obstacle in self.obstacles:
            # check if the agent is at the obstacle position before moving
            if self.agent_position == obstacle.position:
                self.reward = -1
                self.done = True
            obstacle.move(self.grid_size)
            # check if the agent is at the obstacle position after moving
            if self.agent_position == obstacle.position:
                self.reward = -1
                self.done = True

    def _check_goal(self):
        # check if the agent is at the goal position
        if self.agent_position == self.goal_position:
            self.reward = 1
            self.done = True

    def _timeout_check(self):
        # check if timeout is reached
        if self.steps >= self.timeout:
            self.reward = -1
            self.done = True

    def _evaluate_reward(self):
        # define new distance to goal
        new_dist = np.linalg.norm(np.array(self.agent_position) - np.array(self.goal_position))

        # if the agent is moving towards the goal, give a positive reward, if not, give a negative reward
        if new_dist < self.old_dist:
            self.reward = 0.025 * 0.5
        elif new_dist == self.old_dist: #wall hit
            self.reward = -0.05* 0.5
        else:
            self.reward = -0.025* 0.5

        # punish the agent for revisiting old positions
        if self.agent_position in self.last_agent_positions:
            self.reward -= 0.025* 0.5

        # set the new distance to the old distance
        self.old_dist = new_dist

    def _check_goal_reachable(self, goal_position, agent_position, obstacle_positions):
        obstacle_positions = set(tuple(pos) for pos in obstacle_positions)
        check_goal_reachable = a_star_search(agent_position, goal_position, obstacle_positions, self.grid_size)
        if not check_goal_reachable:
            print("Goal is not reachable, resetting environment")
        # if check_goal_reachable:
        # print("Goal is reachable")
        return check_goal_reachable