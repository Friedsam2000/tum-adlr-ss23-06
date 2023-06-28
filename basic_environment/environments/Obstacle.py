class Obstacle:
    """Class to manage Obstacles"""
    def __init__(self, positions, shape, is_moving=False, direction=None):
        self.positions = positions  # list of positions that belong to this obstacle
        self.shape = shape  # the shape of the obstacle
        self.is_moving = is_moving
        self.direction = direction

    def move(self, grid_size):
        """
        Moves the obstacle in the defined direction within the grid size limits
        """
        if self.is_moving:
            min_x = min(position[0] for position in self.positions)
            min_y = min(position[1] for position in self.positions)
            max_x = max(position[0] for position in self.positions)
            max_y = max(position[1] for position in self.positions)

            if self.direction == 0 and min_y > 0:  # up
                self.positions = [[pos[0], pos[1] - 1] for pos in self.positions]
            elif self.direction == 1 and max_y < grid_size[1] - 1:  # down
                self.positions = [[pos[0], pos[1] + 1] for pos in self.positions]
            elif self.direction == 2 and min_x > 0:  # left
                self.positions = [[pos[0] - 1, pos[1]] for pos in self.positions]
            elif self.direction == 3 and max_x < grid_size[0] - 1:  # right
                self.positions = [[pos[0] + 1, pos[1]] for pos in self.positions]
            else:  # If obstacle is at boundary, reverse direction
                # if The obstacle hit a wall, reverse direction
                if self.direction == 0:  # up
                    self.direction = 1
                elif self.direction == 1:  # down
                    self.direction = 0
                elif self.direction == 2:  # left
                    self.direction = 3
                elif self.direction == 3:  # right
                    self.direction = 2


