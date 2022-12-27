from collections import namedtuple, deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import ai_Network as network
import math

tdqnSettings_ = namedtuple("tdqnSettings", ["gamma", "epsilonStart", "epsilonEnd", "epsilonDecay", "capacity",
                                            "learningRate", "targetUpdateFrequency", "batchSize"])

networkSettings_ = namedtuple("networkSettings", ["inputLayerSize", "hiddenLayerSize",
                                                  "outputLayerSize", "dropout"])

optimSettings_ = namedtuple("optimSettings", ["L2Factor"])



class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def Push(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))

    def Sample(self, batchSize):
        return zip(*random.sample(self.memory, batchSize))

    def __len__(self):
        return len(self.memory)



class TDQNAgent():
    
    def __init__(self, tdqnSettings, networkSettings, optimSettings):

        #random seed
        self.tdqnSettings = tdqnSettings
        self.networkSettings = networkSettings
        self.optimSettings = optimSettings
        self.device = torch.device('cuda:'+str(0) if torch.cuda.is_available() else 'cpu')
        self.iteration = 0

        self.ReplayMemory = ReplayMemory(self.tdqnSettings.capacity)

        self.PolicyNetwork = network.FCFFN(*self.networkSettings).to(self.device)
        self.TargetNetwork = network.FCFFN(*self.networkSettings).to(self.device)
        self.TargetNetwork.load_state_dict(self.PolicyNetwork.state_dict())
        self.PolicyNetwork.eval()
        self.TargetNetwork.eval()

        self.optimizer = optim.Adam(self.PolicyNetwork.parameters(), lr=self.tdqnSettings.learningRate, weight_decay=self.optimSettings.L2Factor)

        self.epsilonValue = self.GetEpsilon()


    def GetEpsilon(self):
        st = self.tdqnSettings
        self.epsilonValue = st.epsilonEnd + (st.epsilonStart - st.epsilonEnd) * math.exp(-1 * self.iteration / st.epsilonDecay)
        return self.epsilonValue

    def LearnFromMemory(self):
        
        if len(self.ReplayMemory) < self.tdqnSettings.batchSize:
            return

        self.PolicyNetwork.train()

        state, action, reward, nextState, done = self.ReplayMemory.Sample(self.tdqnSettings.batchSize)

        



        
