import numpy as np
import gymnasium
import torch
from supervisedExtractor.cnnExtractor import CNNExtractor
from environments.GridEnvironment import GridEnvironment

class FeatureExtractedEnv(gymnasium.Wrapper):
    def __init__(self, env=None):
        if env is None:
            env = GridEnvironment(num_obstacles=0, num_frames_to_stack=1)  # default environment
        super().__init__(env)
        # Load the model
        self.pretrained_model = CNNExtractor()
        model_path = '../supervisedExtractor/model.pth'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()

        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4+49,), dtype=np.float32)

    def extract_features(self, observation):

        # Preprocess the observation
        image = observation.astype('float32') / 255.0  # Convert to float and scale values to range [0, 1]
        image = torch.from_numpy(image).permute(2, 0, 1)  # Convert to tensor and permute dimensions
        image = image.unsqueeze(0).to(self.device)

        # Extract features
        predicted_grid, predicted_pos = self.pretrained_model(image)

        # Convert to numpy
        predicted_pos = predicted_pos.cpu().detach().numpy()
        predicted_grid = predicted_grid.cpu().detach().numpy()

        # Get true position
        frame_info = self.env.get_current_frame_info()

        # print("predicted agent position: ", np.round(predicted_pos[0,0:2]))
        # print("true agent position     : ", frame_info['agent_position'])
        #
        # print("predicted goal position : ", np.round(predicted_pos[0,2:4]))
        # print("true goal position      : ", frame_info['goal_position'])

        # combine observations (grid and positions)
        extracted_features = np.concatenate((predicted_pos[0], predicted_grid[0]), axis=0)

        return extracted_features

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.extract_features(obs), reward, terminated, truncated, info

    def reset(self, seed=None):
        obs, info = self.env.reset(seed)
        return self.extract_features(obs), info
