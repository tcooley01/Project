import random as rand
from collections import namedtuple
import torch

Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'reward'])

class ReplayMemory():
    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return rand.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

