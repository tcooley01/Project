import sys
import os
import torch
import torch.nn as nn
from torch.distributions import Categorical
if os.path.join(os.getcwd(), "..", '..', '..', '..') not in sys.path:
    sys.path.append(os.path.join(os.getcwd(), "..", '..', '..', '..'))
from Project.src.components import ReplayMemory as rm

class ActorCritic(nn.Module):
    def __init__(self, state_dims, action_dims, *args, **kwargs):
        super(ActorCritic, self).__init__() 
        self.policy = None
        self.value = None

    def sample(self, state):
        with torch.no_grad():
            action_distribution = self.policy(state)
            value = self.value(state)
        action_distribution = Categorical(action_distribution)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)

        return action, value, log_prob

    def evaluate(self, state, action):
        action_dist = self.policy(state)
        value = self.value(state)
        action_dist = Categorical(action_dist)
        log_probs = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return value, log_probs, entropy




        


