import random as rand
import torch


class EpisodeMemory():
    def __init__(self, num_episodes):
        self.num_episodes = num_episodes
        self.episode_idx = 0
        self.initialized = False
        self.episodes = None
    def add_obs(self, obs, action, reward, next_obs):
        if not self.initialized:
            pass


