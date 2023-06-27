class Obstacle:
    """Class to manage Obstacles"""
    def __init__(self, position, is_moving=False, direction=None):
        self.position = position
        self.is_moving = is_moving
        self.direction = direction

    def move(self, grid_size):
        """
        Moves the obstacle in the defined direction within the grid size limits
        """
        if self.is_moving:
            old_position = self.position.copy()

            if self.direction == 0:  # up
                self.position[1] = max(0, self.position[1] - 1)
            elif self.direction == 1:  # down
                self.position[1] = min(grid_size[1] - 1, self.position[1] + 1)
            elif self.direction == 2:  # left
                self.position[0] = max(0, self.position[0] - 1)
            elif self.direction == 3:  # right
                self.position[0] = min(grid_size[0] - 1, self.position[0] + 1)

            # if The obstacle hit a wall, reverse direction
            if self.position == old_position:
                if self.direction == 0:  # up
                    self.direction = 1
                elif self.direction == 1:  # down
                    self.direction = 0
                elif self.direction == 2:  # left
                    self.direction = 3
                elif self.direction == 3:  # right
                    self.direction = 2

