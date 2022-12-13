from collections import namedtuple, deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import ai_Network as network

tdqnSettings = namedtuple("tdqnSettings", ["gamma", "epsilonStart", "epsilonEnd", "epsilonDecay", "capacity",
                                            "learningRate", "targetUpdateFrequency", "batch_size"]) 

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
    
    def __init__(self, tdqnSettings_, networkSettings_):

        self.tdqnSettings_ = tdqnSettings_
        self.networkSettings_ = networkSettings_
        self.device = torch.device('cuda:'+str(0) if torch.cuda.is_available() else 'cpu')

        self.ReplayMemory = ReplayMemory(self.tdqnSettings_.capacity)

        self.PolicyNetwork = network.FCFFN(*self.networkSettings_).to(self.device)
        self.TargetNetwork = network.FCFFN(*self.networkSettings_).to(self.device)
        self.TargetNetwork.load_state_dict(self.PolicyNetwork.state_dict())
        self.PolicyNetwork.eval()
        self.TargetNetwork.eval()