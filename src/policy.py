import torch
import torch.nn as nn
class Policy_Network(nn.Module):
    def __init__(self, obs_dim, hidden_dim, *args, **kwargs):
        super(Policy_Network, self).__init__()
        self.policy = 
