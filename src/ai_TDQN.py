from collections import namedtuple, deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import ai_Network as network
import math
import numpy as np

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
    
    def __init__(self, TradingEnvironment, tdqnSettings, networkSettings, optimSettings):

        #random seed
        self.TradingEnvironment = TradingEnvironment
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

    def DataPreProcessing(self):
        close = self.TradingEnvironment.dataFrame.Close.values
        low = self.TradingEnvironment.dataFrame.Low.values
        high = self.TradingEnvironment.dataFrame.High.values
        volume = self.TradingEnvironment.dataFrame.Volume.values

        normalizationCoefficients = {"Returns": [], "DeltaPrice": [], "HighLow": [0,1], "Volume": []}
        margin = 1

        returns = [abs((close[i]-close[i-1])/close[i-1]) for i in range(1, len(close))]
        normalizationCoefficients["Returns"] = [0, np.max(returns)*margin]

        deltaPrice = [abs(high[i]-low[i]) for i in range(len(low))]
        normalizationCoefficients["DeltaPrice"] = [0, np.max(deltaPrice)*margin]

        normalizationCoefficients["Volume"] = [np.min(volume)/margin, np.max(volume)*margin]

        return normalizationCoefficients

    def StateProcessing(self, state):
        coeffs = self.DataPreProcessing()

        closePrices = [state[0][i] for i in range(len(state[0]))]
        lowPrices = [state[1][i] for i in range(len(state[1]))]
        highPrices = [state[2][i] for i in range(len(state[2]))]
        volumes = [state[3][i] for i in range(len(state[3]))]

        # 1. Close price => returns => MinMax normalization
        returns = [(closePrices[i]-closePrices[i-1])/closePrices[i-1] for i in range(1, len(closePrices))]
        if coeffs["Returns"][0] != coeffs["Returns"][1]:
            state[0] = [((x - coeffs["Returns"][0])/(coeffs["Returns"][1] - coeffs["Returns"][0])) for x in returns]
        else:
            state[0] = [0 for x in returns]
        # 2. Low/High prices => Delta prices => MinMax normalization
        deltaPrice = [abs(highPrices[i]-lowPrices[i]) for i in range(1, len(lowPrices))]
        if coeffs["DeltaPrice"][0] != coeffs["DeltaPrice"][1]:
            state[1] = [((x - coeffs["DeltaPrice"][0])/(coeffs["DeltaPrice"][1] - coeffs["DeltaPrice"][0])) for x in deltaPrice]
        else:
            state[1] = [0 for x in deltaPrice]
            print("Zero State")
        # 3. Close/Low/High prices => Close price position => No normalization required
        closePricePosition = []
        for i in range(1, len(closePrices)):
            deltaPrice = abs(highPrices[i]-lowPrices[i])
            if deltaPrice != 0:
                item = abs(closePrices[i]-lowPrices[i])/deltaPrice
            else:
                item = 0.5
            closePricePosition.append(item)
        if coeffs["HighLow"][0] != coeffs["HighLow"][1]:
            state[2] = [((x - coeffs["HighLow"][0])/(coeffs["HighLow"][1] - coeffs["HighLow"][0])) for x in closePricePosition]
        else:
            state[2] = [0.5 for x in closePricePosition]
        # 4. Volumes => MinMax normalization
        volumes = [volumes[i] for i in range(1, len(volumes))]
        if coeffs["Volume"][0] != coeffs["Volume"][1]:
            state[3] = [((x - coeffs["Volume"][0])/(coeffs["Volume"][1] - coeffs["Volume"][0])) for x in volumes]
        else:
            state[3] = [0 for x in volumes]

       
        # Process the state structure to obtain the appropriate format
        state = [item for sublist in state for item in sublist]

        return state




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