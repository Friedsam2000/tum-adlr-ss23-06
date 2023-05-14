import numpy as np
from GridEnvironment import GridEnvironment
import cv2

block_size = 50  # size of a block in pixels
block_color = (255, 255, 255)  # white color in BGR
grid_color = (0, 0, 0)  # black color in BGR

env = GridEnvironment(grid_size=(10, 10))

def draw_state(state):
    img = np.zeros((state.shape[0]*block_size, state.shape[1]*block_size, 3), np.uint8)
    for y in range(state.shape[0]):
        for x in range(state.shape[1]):
            if state[y, x] == 1:
                cv2.rectangle(img, (x*block_size, y*block_size), ((x+1)*block_size, (y+1)*block_size), block_color, -1)
    return img

# Display the image in a window
cv2.imshow('Grid', draw_state(env.get_state()))
cv2.waitKey(0)

# Event loop
while True:
    key = cv2.waitKey(0)
    if key == 82:   # up
        action = 0
    elif key == 84: # down
        action = 1
    elif key == 81: # left
        action = 2
    elif key == 83: # right
        action = 3
    elif key == 113:  # 'q' for quit
        break
    else:
        continue

    # Apply the action to the environment
    state = env.step(action)

    # Redraw the state
    cv2.imshow('Grid', draw_state(state))

cv2.destroyAllWindows()
