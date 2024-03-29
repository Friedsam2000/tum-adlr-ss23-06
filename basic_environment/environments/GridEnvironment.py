from collections import deque
import numpy as np
import cv2
import gymnasium
from gymnasium import spaces
from utility.CheckGoalReachable import a_star_search
from .Obstacle import Obstacle
import random


class GridEnvironment(gymnasium.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    action_space = spaces.Discrete(4)

    def __init__(self, grid_size=(24, 24), img_size=(96, 96), render_size=(480, 480), num_last_agent_pos=100,
                 draw_last_agent_pos_in_obs=False, num_frames_to_stack=4, grey_scale=False, render_greyscale=False,
                 num_obstacles=6, size_grid_frame_info=11):
        super().__init__()
        self.render_greyscale = render_greyscale
        self.grey_scale = grey_scale
        self.num_frames_to_stack = num_frames_to_stack
        self.frame_stack = deque(maxlen=num_frames_to_stack)
        self._assert_sizes(grid_size, img_size, render_size)
        self.draw_last_agent_pos_in_obs = draw_last_agent_pos_in_obs
        self.num_obstacles = num_obstacles

        self.size_grid_frame_info = size_grid_frame_info

        self.num_last_agent_pos = num_last_agent_pos
        self.grid_size = grid_size
        self.img_size = img_size
        self.render_size = render_size

        if self.grey_scale:
            self.observation_space = spaces.Box(low=0, high=255, shape=(img_size[0], img_size[1], num_frames_to_stack),
                                                dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(img_size[0], img_size[1], num_frames_to_stack * 3),
                                                dtype=np.uint8)

    def reset(self, seed=None):
        # Reset the environment and optionally set the random seed
        if seed is not None:
            np.random.seed(seed)

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

        return self._getObservation(), {}

    def step(self, action):
        self.steps += 1

        # Move the agent
        self._move_agent(action)

        # Check if the agent hit an obstacle
        if self._check_obstacle_collision():
            return self._getObservation(), -1, True, False, {}

        # Move the obstacles
        self._move_obstacles()

        # Check if an obstacle hit the agent
        if self._check_obstacle_collision():
            return self._getObservation(), -1, True, False, {}

        # Check if the agent reached the goal
        if self._check_goal():
            return self._getObservation(), 1, True, False, {}

        # Check if the agent reached the timeout
        if self._timeout_check():
            return self._getObservation(), -1, False, True, {}

        # Evaluate the reward if the agent did not reach the goal or a timeout or hit an obstacle
        return self._getObservation(), self._evaluate_reward(), False, False, {}

    def _getObservation(self):
        if self.grey_scale:
            return np.array(self.convertGreyscale(self.getImg()), dtype=np.uint8)
        else:
            return np.array(self.getImg(), dtype=np.uint8)

    def render(self, mode='human'):
        # Get the last frame from the deque
        img = self.getImg().copy()  # Create a copy of the image to avoid modifying the original

        # Get the current agent position
        current_pos = [int(self.agent_position[0] * self.img_size[0] / self.grid_size[0]),
                       int(self.agent_position[1] * self.img_size[1] / self.grid_size[1])]

        # Draw the old agent positions on the copied image if they are not in the observation
        if not self.draw_last_agent_pos_in_obs:
            for old_pos in self.last_agent_positions:
                if old_pos[0] != -1 and old_pos[1] != -1:  # only draw if it's not the initial [-1, -1]
                    scaled_old_pos = [int(old_pos[0] * self.img_size[0] / self.grid_size[0]),
                                      int(old_pos[1] * self.img_size[1] / self.grid_size[1])]

                    # Check if the old position is not the current position
                    if scaled_old_pos != current_pos:
                        # Create a kernel filled with the desired color
                        # kernel size is the multiple of the grid size and the image size
                        kernel_size = int(self.img_size[0] / self.grid_size[0])
                        color_kernel = np.ones((kernel_size, kernel_size, 3)) * np.array([255, 255, 0])
                        # Replace the corresponding area in the image with the color kernel
                        img[scaled_old_pos[0]:scaled_old_pos[0] + kernel_size,
                        scaled_old_pos[1]:scaled_old_pos[1] + kernel_size,
                        -3:] = color_kernel

        # Draw the obstacles on the copied image over old agent positions
        for obstacle in self.obstacles:
            for pos in obstacle.positions:
                scaled_pos = [int(pos[0] * self.img_size[0] / self.grid_size[0]),
                              int(pos[1] * self.img_size[1] / self.grid_size[1])]
                kernel_size = int(self.img_size[0] / self.grid_size[0])
                color_kernel = np.ones((kernel_size, kernel_size, 3)) * np.array([0, 0, 255])  # Blue for obstacles
                img[scaled_pos[0]:scaled_pos[0] + kernel_size, scaled_pos[1]:scaled_pos[1] + kernel_size,
                -3:] = color_kernel

        if self.render_greyscale:
            # convert rgb to greyscale
            display_img = self.convertGreyscale(img)
            # Use only the newest frame for visualization
            display_img = display_img[:, :, -1:]
        else:
            # Use the newest 3 channels for displaying
            display_img = img[:, :, -3:]

        # Resize the image for better visualization
        display_img = cv2.resize(display_img, self.render_size, interpolation=cv2.INTER_NEAREST)

        # Plot symbols at the end of each rendered episode
        if self._check_goal():
            self.draw_checkmark(display_img)
            cv2.waitKey(200)
        elif self._timeout_check() or self._check_obstacle_collision():
            self.draw_cross(display_img)
            cv2.waitKey(200)


        # Display the image
        cv2.imshow('image', display_img)
        cv2.waitKey(80)

    def close(self):
        cv2.destroyAllWindows()

    def getImg(self):
        # Create a base image to represent the grid
        base_image = np.zeros((self.grid_size[0], self.grid_size[1], 3))

        if self.draw_last_agent_pos_in_obs:
            # Draw the old agent positions in the observation
            for old_pos in self.last_agent_positions:
                if old_pos[0] != -1 and old_pos[1] != -1:  # only draw if it's not the initial [-1, -1]
                    base_image[old_pos[0], old_pos[1]] = [255, 255, 0]  # Yellow for old agent positions

        # Draw the agent, goal and obstacles on the base image
        # assuming the agent, goal and obstacles are represented as different colors in RGB
        base_image[self.agent_position[0], self.agent_position[1]] = [255, 0, 0]  # blue for agent
        base_image[self.goal_position[0], self.goal_position[1]] = [0, 255, 0]  # Green for goal
        for obstacle in self.obstacles:
            for pos in obstacle.positions:
                base_image[pos[0], pos[1]] = [0, 0, 255]  # red for obstacles

        # Scale the image up for the network
        scaled_image = cv2.resize(base_image, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_AREA)

        # Add the new frame to the stack
        self.frame_stack.append(scaled_image)

        # Concatenate the frames along the channel dimension
        stacked_frames = np.concatenate(self.frame_stack, axis=-1)

        # Return the stacked frames as a numpy array
        return stacked_frames

    def convertGreyscale(self, stacked_frames):
        # Assuming the input is 96x96x12, split the frames into 4 frames each having 3 color channels
        color_frames = np.split(stacked_frames, 4, axis=2)

        # Initialize an empty list for the greyscale frames
        greyscale_frames = []

        # Loop through each color frame
        for frame in color_frames:
            # Convert the depth of the image to 8-bit
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            # Split the frame into B, G, R channels
            b, g, r = cv2.split(frame)

            # Compute a weighted sum of the B, G, R channels, with conditional weighting
            greyscale_frame = np.zeros_like(b)

            # If blue is dominant
            greyscale_frame[b > r] = 0.66 * b[b > r]

            # If green is dominant
            greyscale_frame[g > r] = 1 * g[g > r]

            # If red is dominant
            greyscale_frame[r > b] = 0.33 * r[r > b]

            # Add the greyscale frame to the list
            greyscale_frames.append(greyscale_frame)

        # Stack the frames
        stacked_frames = np.stack(greyscale_frames, axis=2)

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
        # Define various obstacle shapes as 2D binary matrices
        shapes = [
            np.array([[1, 1], [1, 0]]),  # L shape
            np.array([[1]]),  # Single cell
            np.array([[1, 1, 1, 1]]),  # Straight line in x direction
            np.array([[1], [1], [1], [1]]),  # Straight line in y direction
            np.array([[1, 1], [1, 1]]),  # 2x2 square
            np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]),  # Diamond shape
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Diagonal line
        ]
        goal_is_reachable = False
        while not goal_is_reachable:
            self.obstacles = []  # Clear any existing obstacles

            # Choose a random number of obstacles to generate
            num_obstacles = self.num_obstacles

            for _ in range(num_obstacles):
                # Randomly select a shape for this obstacle
                shape = random.choice(shapes)
                # Generate the positions occupied by this obstacle
                obstacle_positions = self._generate_obstacle_block(shape)

                # Check if the obstacle is within 3 grid cells of the agent
                is_moving = not any(
                    abs(pos[0] - self.agent_position[0]) < 2 and abs(pos[1] - self.agent_position[1]) < 3
                    for pos in obstacle_positions
                )

                direction = np.random.choice([0, 1, 2, 3]) if is_moving else None
                self.obstacles.append(Obstacle(obstacle_positions, shape, is_moving, direction))

            # Flatten the obstacle positions into a single list
            self.obstacle_positions = [position for obstacle in self.obstacles for position in obstacle.positions]

            # Assuming the goal is reachable for moving obstacles
            goal_is_reachable = True

    def _generate_obstacle_block(self, shape):
        # Get the dimensions of the shape
        height, width = shape.shape

        while True:
            # Randomly choose the top-left position of this obstacle
            obstacle_position = [np.random.randint(0, self.grid_size[0] - (width - 1)),
                                 np.random.randint(0, self.grid_size[1] - (height - 1))]

            # Check if this obstacle collides with the agent or the goal
            collision = False
            obstacle_positions = []
            for i in range(width):
                for j in range(height):
                    if shape[j, i] == 1:
                        pos = [obstacle_position[0] + i, obstacle_position[1] + j]
                        # Check collision with agent or goal
                        if pos == self.agent_position or pos == self.goal_position:
                            collision = True
                        obstacle_positions.append(pos)
            if not collision:
                return obstacle_positions

    def _check_obstacle_collision(self):
        # check if the agent hit an obstacle
        for obstacle in self.obstacles:
            if self.agent_position in obstacle.positions:
                return True
        return False

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

    def _move_obstacles(self):
        for obstacle in self.obstacles:
            obstacle.move(self.grid_size)

    def _check_goal(self):
        # check if the agent is at the goal position
        if self.agent_position == self.goal_position:
            return True
        else:
            return False

    def _timeout_check(self):
        # check if timeout is reached
        if self.steps >= self.timeout:
            return True
        else:
            return False

    def _evaluate_reward(self):
        # define new distance to goal
        new_dist = np.linalg.norm(np.array(self.agent_position) - np.array(self.goal_position))

        # if the agent is moving towards the goal, give a positive reward, if not, give a negative reward
        if new_dist < self.old_dist:
            reward = 0.025 * 2
        elif new_dist == self.old_dist:  # wall hit
            reward = -0.05 * 2
        else:
            reward = -0.025 * 2

        # set the new distance to the old distance
        self.old_dist = new_dist

        return reward

    def _check_goal_reachable(self, goal_position, agent_position, obstacle_positions):
        obstacle_positions = set(tuple(pos) for pos in obstacle_positions)
        check_goal_reachable = a_star_search(agent_position, goal_position, obstacle_positions, self.grid_size)
        if not check_goal_reachable:
            print("Goal is not reachable, resetting environment")
        # if check_goal_reachable:
        # print("Goal is reachable")
        return check_goal_reachable

    def get_current_frame_info(self):
        agent_pos = self.agent_position
        goal_pos = self.goal_position

        # Get neighboring cells
        neighbors_content = self.get_neighboring_cells_content(agent_pos)

        return {
            "agent_position": agent_pos,
            "goal_position": goal_pos,
            "neighboring_cells_content": neighbors_content
        }

    def get_neighboring_cells_content(self, position):

        distance = (self.size_grid_frame_info-1) // 2


        neighbors_content = [[0 for _ in range(2 * distance + 1)] for _ in
                             range(2 * distance + 1)]  # Initialize with 0 for 'no_obstacle'

        # Pre-calculate obstacle positions
        obstacle_positions = {(pos[0], pos[1]) for obstacle in self.obstacles for pos in obstacle.positions}

        for dx in range(-distance, distance + 1):
            for dy in range(-distance, distance + 1):
                x, y = position[0] + dx, position[1] + dy
                if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:  # Check bounds
                    # Check if the current cell corresponds to any obstacle's position
                    content = 1 if (x, y) in obstacle_positions else 0  # 'obstacle' if present else 'no_obstacle'
                    # Update the corresponding cell in the neighbors_content grid
                    neighbors_content[dx + distance][dy + distance] = content

        return neighbors_content

    def draw_checkmark(self, img):
        # Define the points for the checkmark
        # You can adjust these points to get the desired size and position
        point1 = (int(img.shape[1] * 0.3), int(img.shape[0] * 0.5))
        point2 = (int(img.shape[1] * 0.5), int(img.shape[0] * 0.7))
        point3 = (int(img.shape[1] * 0.8), int(img.shape[0] * 0.3))
        thickness = 10
        color = (0, 255, 0)  # Green

        cv2.line(img, point1, point2, color, thickness)
        cv2.line(img, point2, point3, color, thickness)

    def draw_cross(self, img):
        # Define the points for the cross
        # Adjust these points for desired size and position
        point1 = (int(img.shape[1] * 0.3), int(img.shape[0] * 0.3))
        point2 = (int(img.shape[1] * 0.8), int(img.shape[0] * 0.7))
        point3 = (int(img.shape[1] * 0.8), int(img.shape[0] * 0.3))
        point4 = (int(img.shape[1] * 0.3), int(img.shape[0] * 0.7))
        thickness = 10
        # make the color pink
        color = (255, 0, 255)  # Pink

        cv2.line(img, point1, point2, color, thickness)
        cv2.line(img, point3, point4, color, thickness)



