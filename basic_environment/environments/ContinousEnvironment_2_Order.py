from collections import deque
import math
import numpy as np
import cv2
import gym
from gym import spaces

class CustomEnv_2order_dyn(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, agent_size=0.2, goal_size=0.8, grid_size=(16,16), img_size=(96, 96), render_size=(960, 960), num_last_agent_pos=2, nr_obstacles=0, nr_goal_pos=15):

        #Super init
        super(CustomEnv_2order_dyn, self).__init__()

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
        self.nr_obstacles = nr_obstacles
        self.nr_goal_pos = nr_goal_pos
        if(nr_obstacles > 0):
            # obstacles
            self.obstacles = np.zeros((nr_obstacles, 3), dtype=np.single)

        #Set of Goal Positions for training
        self.pot_goal_pos = np.zeros((nr_goal_pos,2), dtype=np.single)

        #Sample Set of Goal Positions
        for i in range(0,nr_goal_pos):
            self.pot_goal_pos[i,0] = np.random.uniform(0,self.grid_size[0])
            self.pot_goal_pos[i,1] = np.random.uniform(0,self.grid_size[1])

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.single)
        # Vector input
        self.observation_space = spaces.Box(low=0, high=grid_size[0], shape=(8+nr_obstacles*3,), dtype=np.single)

        # image observation_space
        #self.observation_space = spaces.Box(low=0, high=255, shape=(img_size[0], img_size[1], 3), dtype=np.uint8)

        # max velocity
        self.max_vel = 2.0

        #Agent Mass
        self.agent_mass = 1.0
        self.agent_mass_matrix = np.zeros((2,2), dtype=np.single)
        self.agent_mass_matrix[0,0] = self.agent_mass
        self.agent_mass_matrix[1,1] = self.agent_mass

        #Agent Damping Matrix
        self.agent_damping_matrix = np.zeros((2,2), dtype=np.single)
        self.agent_damping_matrix[0,0] = 0.1
        self.agent_damping_matrix[1,1] = 0.1

        # Number of simulation steps per call of self.step()
        self.nr_sim_steps = 1

        # delta_t per simulation timestep
        self.delta_t = 1 / self.nr_sim_steps

        #Newmark Parameters
        self.gamma = 0.5
        self.beta = 0.25
        self.S = self.agent_mass_matrix + self.beta * self.delta_t * self.agent_damping_matrix
        self.inv_S = np.linalg.inv(self.S)



    def reset(self):
        self.done = False

        # agent velocity
        self.agent_vel = np.zeros((2,), dtype=np.single)
        self.agent_acc = np.zeros((2,), dtype=np.single)

        # set the agent position
        self.agent_position = np.array([np.random.uniform(0,self.grid_size[0]), np.random.uniform(0,self.grid_size[1])], dtype=np.single)
        #self.agent_position = np.array([4.0,4.0], dtype=np.single)


        # set the goal position
        #self.goal_position = np.array([np.random.uniform(0,self.grid_size[0]), np.random.uniform(0,self.grid_size[1])], dtype=np.single)
        #self.goal_position = np.array([10.0,10.0], dtype=np.single)
        goal_pos = np.random.randint(0, self.nr_goal_pos)
        self.goal_position = np.array([self.pot_goal_pos[goal_pos, 0], self.pot_goal_pos[goal_pos, 1]], dtype=np.single)

        # define last distance to goal
        self.initial_dist = np.linalg.norm(self.agent_position - self.goal_position)

        # ensure that goal and start position are sufficiently far apart
        while self.initial_dist < 6 * (self.goal_size + self.agent_size):
            self.agent_position = np.array(
                [np.random.uniform(0, self.grid_size[0]), np.random.uniform(0, self.grid_size[1])], dtype=np.single)
            self.initial_dist = np.linalg.norm(self.agent_position - self.goal_position)

        # store old distance for reward
        self.old_dist = self.initial_dist



        # define step counter and timeout as 6 times the time it takes to travel from start to goal
        self.steps = 0
        self.timeout = 6 * round(self.initial_dist)

        for i in range(0, self.nr_obstacles):
            n = 0
            resample = False
            # create obstacle
            self.obstacles[i,2] = round(np.random.uniform(0.4,0.6),1)
            self.obstacles[i,0] = np.random.uniform(0,self.grid_size[0])
            self.obstacles[i,1] = np.random.uniform(0,self.grid_size[1])
            # check if valid obstacle (not overlapping with other obstacles or goal and start position)
            resample = self.check_obstacle_creation(i)
            # in case of invalid obstacle resample up to 2ß times
            while n < 20 and resample:
                n = n + 1
                self.obstacles[i, 2] = round(np.random.uniform(0.4, 0.6), 1)
                self.obstacles[i, 0] = np.random.uniform(0, self.grid_size[0])
                self.obstacles[i, 1] = np.random.uniform(0, self.grid_size[1])
                resample = self.check_obstacle_creation(i)
                print('Resampeling Obstacle!')
            # if n > 20 leave the remaining obstacles to be creaeted as [0.0,0.0,0.0]
            if n >= 20:
                return self.get_observation()

        # define observation
        observation = self.get_observation()

        return observation

    def step(self, action):
        self.reward = 0
        self.steps += 1

        #Newmark method for linear second order problems for n simulation steps with delta_t per step of RL algorithm
        # -> action is applied n times -> stepwise constant input Force
        for j in range(0, self.nr_sim_steps):
            #Prediction
            pos_star = self.agent_position + self.delta_t * self.agent_vel + (0.5 - self.beta)*pow(self.delta_t,2)*self.agent_acc
            vel_star = self.agent_vel + (1 - self.gamma) * self.delta_t * self.agent_acc
            #Calculation of acceleration (Calculation of S/S^-1 in init/reset since S=const)
            self.agent_acc = np.matmul(self.inv_S, (action - np.matmul(self.agent_damping_matrix, vel_star)))
            #Correction
            self.agent_position = pos_star + pow(self.delta_t,2) * self.beta * self.agent_acc
            self.agent_vel = vel_star + self.delta_t * self.gamma * self.agent_acc

        # observation
        observation = self.get_observation()

        # compute distance
        new_dist = np.linalg.norm(np.array(self.agent_position) - np.array(self.goal_position))
        delta = abs(new_dist - self.old_dist)

        # check if the agent is at the goal position
        if new_dist < self.goal_size:
            self.reward = 1

            self.done = True
            # return info that goal was reached
            return observation, self.reward, self.done, {"goal": True, "obstacle": False}

        # check if upper or left bound of grid is hit
        if (self.agent_position[0] <= 0.0) or (self.agent_position[1] <= 0.0):
            self.reward = -1
            self.done = True
            return observation, self.reward, self.done, {"goal": False, "obstacle": True}

        # check if lower or right bound of grid is hit
        if (self.agent_position[0] >= self.grid_size[0]) or (self.agent_position[1] >= self.grid_size[1]):
            self.reward = -1
            self.done = True
            return observation, self.reward, self.done, {"goal": False, "obstacle": True}

        # check if timeout
        if self.steps >= self.timeout:
            self.reward = -1
            self.done = True
            return observation, self.reward, self.done, {"goal": False, "obstacle": False}

        # check if obstacle is hit
        for i in range(0, self.nr_obstacles):
            distance = np.linalg.norm(self.agent_position - self.obstacles[i,0:2])
            if distance <= self.agent_size + self.obstacles[i,2]:
                self.reward = -1
                self.done = True
                return observation, self.reward, self.done, {"goal": False, "obstacle": True}

        # if the agent is moving towards the goal, give a positive reward, if not, give a negative reward
        #if new_dist < self.old_dist:
            #self.reward = 0.025 * delta
        #else:
            #self.reward = -0.025 * delta

        # set the new distance to the old distance
        self.old_dist = new_dist

        return observation, self.reward, self.done, {}

    def render(self, mode='human'):
        # Render the environment to the screen
        img = self.getImg()
        #img = self.get_observation()
        #img = cv2.resize(self.getImg(), self.img_size, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('image', img)
        cv2.waitKey(200)

    def close(self):
        cv2.destroyAllWindows()

    def get_observation(self):
        observation = np.zeros((8+self.nr_obstacles*3,), dtype=np.single)
        observation[0] = self.agent_position[0]
        observation[1] = self.agent_position[1]
        observation[2] = self.agent_vel[0]
        observation[3] = self.agent_vel[1]
        observation[4] = self.agent_size
        observation[5] = self.goal_position[0]
        observation[6] = self.goal_position[1]
        observation[7] = self.goal_size
        for i in range(0, self.nr_obstacles):
            observation[8+i*3]=self.obstacles[i,0]
            observation[8+(i*3)+1]=self.obstacles[i,1]
            observation[8+(i*3)+2]=self.obstacles[i,2]

        #observation = cv2.resize(self.getImg(), self.img_size, interpolation=cv2.INTER_NEAREST)
        return observation

    def getImg(self):
        agent_color = (255, 255, 255)
        goal_color = (0, 255, 0)
        obstacle_color = (0, 0, 255)
        old_agent_color = (255, 192, 203)

        img = np.zeros((self.grid_size[0], self.grid_size[1], 3), dtype=np.uint8)

        # scale the grid to render_size
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

        # plot obstacles
        for i in range(0, self.nr_obstacles):
            x_obs = round(self.obstacles[i,0]*self.scaling[0])
            y_obs = round(self.obstacles[i,1] * self.scaling[1])
            radius_obs = math.ceil(self.obstacles[i,2] * self.scaling[0])
            img = cv2.circle(img, (x_obs, y_obs), radius_obs, obstacle_color, -1)
            
        return img

    def check_obstacle_creation(self, index):
        distance_goal = np.linalg.norm(self.goal_position - self.obstacles[index,0:2])
        distance_agent = np.linalg.norm(self.agent_position - self.obstacles[index,0:2])
        # ensure that obstacle is neither in goal nor agent position
        if distance_goal <= self.goal_size + self.obstacles[index,2] or distance_agent <= self.agent_size + self.obstacles[index,2]:
            return True
        if index == 0:
            return False
        for i in range(0, index):
            distance_obstacle = np.linalg.norm(self.obstacles[i,0:2] - self.obstacles[index,0:2])
            # ensure obstacles have a 1.5 times the agent size gap in between so no area of grid is blocked
            if distance_obstacle <= self.obstacles[i,2] + self.obstacles[index,2] + (1.5 * self.agent_size):
                return True

        return False

