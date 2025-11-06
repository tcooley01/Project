import random as rand
from collections import deque
from collections import namedtuple
import torch

transition = namedtuple('Transition', ['state', 'action', 'next_state', 'reward'])

class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque([], maxlen = capacity)

    def push(self, *args):
        self.memory.append(transition(*args))
    
    def sample(self, batch_size):
        return rand.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

