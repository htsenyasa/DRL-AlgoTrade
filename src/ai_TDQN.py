from collections import namedtuple, deque
import random


class ReplayMemory():
    transition = namedtuple("transition", ["state", "action", "reward", "nextState", "done"])

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class TDQNAgent():
    ...