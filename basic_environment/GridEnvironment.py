import numpy as np

class GridEnvironment:
    def __init__(self, grid_size=(10, 10)):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.block_position = [self.grid_size[0] // 2, self.grid_size[1] // 2]
        return self.get_state()

    def step(self, action):
        if action == 0:   # up
            self.block_position[1] = max(0, self.block_position[1] - 1)
        elif action == 1: # down
            self.block_position[1] = min(self.grid_size[1] - 1, self.block_position[1] + 1)
        elif action == 2: # left
            self.block_position[0] = max(0, self.block_position[0] - 1)
        elif action == 3: # right
            self.block_position[0] = min(self.grid_size[0] - 1, self.block_position[0] + 1)

        return self.get_state()

    def get_state(self):
        state = np.zeros(self.grid_size)
        state[self.block_position[1], self.block_position[0]] = 1
        return state
