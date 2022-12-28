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
                                            "learningRate", "targetUpdateFrequency", "batchSize", "gradientClipping", "targetNetworkUpdate"])

networkSettings_ = namedtuple("networkSettings", ["inputLayerSize", "hiddenLayerSize",
                                                  "outputLayerSize", "dropout"])

optimSettings_ = namedtuple("optimSettings", ["L2Factor"])

class MemoryElement():
    def __init__(self, device, *args):
        self.state = torch.tensor(args[0], dtype=torch.float, device=device)
        self.action = torch.tensor(args[1], dtype=torch.long, device=device)
        self.reward = torch.tensor(args[2], dtype=torch.float, device=device)
        self.nextState = torch.tensor(args[3], dtype=torch.float, device=device)
        self.done = torch.tensor(args[4], dtype=torch.float, device=device)


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

        state = torch.tensor(state, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        nextState = torch.tensor(nextState, dtype=torch.float, device=self.device)
        done = torch.tensor(done, dtype=torch.float, device=self.device)

        currentQValues = self.policyNetwork(state).gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            nextActions = torch.max(self.policyNetwork(nextState), 1).indices
            nextQValues = self.targetNetwork(nextState).gather(1, nextActions.unsqueeze(1)).squeeze(1)
            expectedQValues = self.tdqnSettings.reward + self.tdqnSettings.gamma * nextQValues * (1 - done)

        loss = F.smooth_l1_loss(currentQValues, expectedQValues)
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policyNetwork.parameters(), self.tdqnSettings.gradientClipping)
        self.optimizer.step()
        
        if(self.iterations % self.tdqnSettings.targetNetworkUpdate == 0):
            self.TargetNetwork.load_state_dict(self.PolicyNetwork.state_dict())
        
        self.PolicyNetwork.eval()

    def Training(self):
        ...