import csv
import cv2
import sys
sys.path.append("..")  # noqa: E402
from environments.GridEnvironment import GridEnvironment
import os
import logging
from multiprocessing import Pool

logging.basicConfig(level=logging.INFO)

def write_image(image_data):
    name, img = image_data
    cv2.imwrite(f"test_images/{name}", img)


# clear or create test_images directory
if os.path.exists("test_images"):
    for file in os.listdir("test_images"):
        os.remove(f"test_images/{file}")
else:
    os.mkdir("test_images")

#delete labels.csv if it exists
if os.path.exists("labels.csv"):
    os.remove("labels.csv")

# Define the field names for the CSV file
fieldnames = ['image_name', 'agent_pos_x', 'agent_pos_y', 'goal_pos_x', 'goal_pos_y'] + [f'neighbor_{i}_{j}' for i in range(7) for j in range(7)]

# Create the environment
env = GridEnvironment(num_last_agent_pos=0,num_obstacles=0, num_frames_to_stack=1)

# Reset the environment
obs, info = env.reset()

# Prepare for batching
batch_size = 1000
image_batch = []  # Initialize as an empty list
csv_rows = []  # Initialize as an empty list
batch_index = 0

# Open CSV file outside the loop
with open('labels.csv', 'a', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()  # Write the header only once

    episode = 0
    timestep = 0
    num_images = 100000
    while episode < num_images:

        # Reset the environment
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode += 1
        timestep = 0

        # Extract the current frame info
        current_frame_info = env.get_current_frame_info()
        agent_pos = current_frame_info["agent_position"]
        goal_pos = current_frame_info["goal_position"]
        neighbors_content = current_frame_info["neighboring_cells_content"]

        # Extract the last RGB image
        image_name = f"episode_{episode}_timestep_{timestep}.png"
        last_rgb_image = obs
        image_batch.append((image_name, last_rgb_image))  # Append to image_batch

        # Prepare row data for CSV
        row_data = {
            'image_name': image_name,
            'agent_pos_x': agent_pos[0],
            'agent_pos_y': agent_pos[1],
            'goal_pos_x': goal_pos[0],
            'goal_pos_y': goal_pos[1]
        }
        row_data.update({f'neighbor_{i}_{j}': neighbors_content[i][j] for i in range(7) for j in range(7)})
        csv_rows.append(row_data)  # Append to csv_rows

        batch_index += 1

        # Write images and CSV rows in batches
        if batch_index == batch_size:
            with Pool() as pool:
                pool.map(write_image, image_batch)
            writer.writerows(csv_rows)
            batch_index = 0  # Reset the index for the next batch
            image_batch = []  # Reset image_batch
            csv_rows = []  # Reset csv_rows

        #Print progress as percentage
        progress_percentage = (episode / num_images) * 100
        logging.info(f"Current episode: {episode}, Progress: {progress_percentage:.2f}%")


    # Write any remaining images and rows outside the loop
    if image_batch:  # Check if there are any remaining images
        with Pool() as pool:
            pool.map(write_image, image_batch)
    if csv_rows:  # Check if there are any remaining rows
        writer.writerows(csv_rows)