import sys
import os
import torch
import torch.nn as nn
from torch.distributions import Categorical
if os.path.join(os.getcwd(), "..", '..', '..', '..') not in sys.path:
    sys.path.append(os.path.join(os.getcwd(), "..", '..', '..', '..'))
from Project.src.components import ReplayMemory as rm

class PolicyValue(nn.Module):
    def __init__(self, state_dims, action_dims, *args, **kwargs):
        super(PolicyValue, self).__init__() 
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

class PPO():
    def __init__(self,
                 state_dims,
                 action_dims, 
                 eps = 0.1, 
                 beta = 0.05):
        self.policy = PolicyValue(state_dims, action_dims)
        self.old_policy = PolicyValue(state_dims, action_dims)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.eps = eps
        self.beta = beta
        self.replay_memory = rm.ReplayMemory()
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr = 1e-3)#this is highly highly hacky needs fixing
        self.value_loss = nn.MSELoss()
        
    def push(self, obs):
        self.replay_memory.push(obs)
    
    def pull(self):
        return self.replay_memory.sample()
    
    def make_action(self, state):
        action, value, log_prob = self.old_policy.sample(state)
        return action, value, log_prob
    def train(self):
        pass
