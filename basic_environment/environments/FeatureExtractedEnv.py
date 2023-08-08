import numpy as np
import gymnasium
import torch
from supervisedExtractor.cnnExtractor import CNNExtractor
from environments.GridEnvironment import GridEnvironment

class FeatureExtractedEnv(gymnasium.Wrapper):
    def __init__(self, env=None):
        if env is None:
            env = GridEnvironment(num_obstacles=0, num_frames_to_stack=4, size_grid_frame_info=11)  # default environment
        super().__init__(env)
        self.num_frames_to_stack = env.num_frames_to_stack
        # Load the model
        self.pretrained_model = CNNExtractor()
        model_path = '../supervisedExtractor/model.pth'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()

        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2*self.num_frames_to_stack, env.size_grid_frame_info, env.size_grid_frame_info),
            dtype=np.float32)

    def extract_features(self, observation):
        frame_dim = observation.shape[2] // self.num_frames_to_stack
        extracted_features_list = []

        for i in range(self.num_frames_to_stack):
            current_frame = observation[:, :, i * frame_dim:(i + 1) * frame_dim]

            # Preprocess the current frame
            image = current_frame.astype('float32') / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
            image = image.unsqueeze(0).to(self.device)

            # Extract features
            predicted_grid, predicted_pos = self.pretrained_model(image)

            # Convert to numpy
            predicted_pos = predicted_pos.cpu().detach().numpy()

            # Normalize the predicted position
            predicted_pos = predicted_pos / 23.0

            predicted_grid = predicted_grid.cpu().detach().numpy()

            # Convert to binary grid
            predicted_grid = (predicted_grid > 0.5).astype(float)

            # Reshape the 2x2 predicted position to self.env.size_grid_frame_info^2 with zeros
            position_frame = np.zeros((self.env.size_grid_frame_info, self.env.size_grid_frame_info))

            # Assign the predicted position to the center of the frame
            position_frame[2:4, 2:4] = predicted_pos[0].reshape(2, 2)  # Modified line

            predicted_grid_reshaped = predicted_grid[0].reshape(self.env.size_grid_frame_info, self.env.size_grid_frame_info)
            extracted_frame_features = np.stack([predicted_grid_reshaped, position_frame], axis=0)

            extracted_features_list.append(extracted_frame_features)

        # Stack features from all frames
        extracted_features = np.concatenate(extracted_features_list, axis=0)

        return extracted_features

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.extract_features(obs), reward, terminated, truncated, info

    def reset(self, seed=None):
        obs, info = self.env.reset(seed)
        return self.extract_features(obs), info
