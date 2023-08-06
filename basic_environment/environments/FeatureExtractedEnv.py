import numpy as np
import gymnasium
import torch
from supervisedExtractor.cnnExtractor import CNNExtractor
from environments.GridEnvironment import GridEnvironment
from torchvision import transforms


class FeatureExtractedEnv(gymnasium.Wrapper):
    def __init__(self, env=None):
        if env is None:
            env = GridEnvironment(num_obstacles=0)  # default environment
        super().__init__(env)
        # Load the model
        self.pretrained_model = CNNExtractor()
        model_path = '../supervisedExtractor/model.pth'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()



        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def extract_features(self, observation):
        # If the observation is a numpy array
        observation = observation[:, :, 9:12]
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        reshaped_observation = transform(observation)
        reshaped_observation = reshaped_observation.unsqueeze(0).to(self.device)

        # Extract features
        predicted_grid, predicted_pos = self.pretrained_model(reshaped_observation)

        # Get true x_position
        frame_info = self.env.get_current_frame_info()

        predicted_positions = predicted_pos[0].cpu().numpy()

        true_position = np.array([frame_info['x_position'], frame_info['y_position']])

        print("predicted position: ", predicted_positions)
        print("true position: ", true_position)


        # return only position for now
        return predicted_pos.squeeze().cpu().detach().numpy()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.extract_features(obs), reward, terminated, truncated, info

    def reset(self, seed=None):

        obs, info = self.env.reset(seed)
        return self.extract_features(obs), info
