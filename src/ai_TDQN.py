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
import tf_DataAugmentation as da
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler

tdqnSettings_ = namedtuple("tdqnSettings", ["gamma", "epsilonStart", "epsilonEnd", "epsilonDecay", 
                                            "capacity", "learningRate", "targetUpdateFrequency",
                                            "batchSize", "gradientClipping", "targetNetworkUpdate",
                                            "alpha", "numberOfEpisodes", "rewardClipping"])

networkSettings_ = namedtuple("networkSettings", ["inputLayerSize", "hiddenLayerSize", "outputLayerSize",
                                                  "dropout"])

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
    
    def __init__(self, TrainingEnvironment, TestingEnvironment, tdqnSettings, networkSettings, optimSettings):

        #random seed
        self.TrainingEnvironment = TrainingEnvironment
        self.TestingEnvironment = TestingEnvironment
        self.tdqnSettings = tdqnSettings
        self.networkSettings = networkSettings
        self.optimSettings = optimSettings
        self.device = torch.device('cuda:'+str(0) if torch.cuda.is_available() else 'cpu')
        self.iteration = 0
        self.loss = []

        self.ReplayMemory = ReplayMemory(self.tdqnSettings.capacity)

        torch.cuda.manual_seed_all(10)
        self.PolicyNetwork = network.FCFFN(*self.networkSettings).to(self.device)
        self.TargetNetwork = network.FCFFN(*self.networkSettings).to(self.device)
        self.TargetNetwork.load_state_dict(self.PolicyNetwork.state_dict())
        self.PolicyNetwork.eval()
        self.TargetNetwork.eval()

        self.optimizer = optim.Adam(self.PolicyNetwork.parameters(), lr=self.tdqnSettings.learningRate, weight_decay=self.optimSettings.L2Factor)

        self.epsilonValue = self.GetEpsilon(self.iteration)



    def GetEpsilon(self, iteration):
        st = self.tdqnSettings
        self.epsilonValue = st.epsilonEnd + (st.epsilonStart - st.epsilonEnd) * math.exp(-1 * iteration / st.epsilonDecay)
        return self.epsilonValue



    def DataPreProcessing(self, testingFlag = False):

        if testingFlag == True:
            DataAugmentation = da.DataAugmentation()
            env = DataAugmentation.lowPassFilter(self.TrainingEnvironment, 5)
        else:
            env = self.TrainingEnvironment

        close = env.dataFrame.Close.values
        low = env.dataFrame.Low.values
        high = env.dataFrame.High.values
        volume = env.dataFrame.Volume.values

        normalizationCoefficients = {"Returns": [], "DeltaPrice": [], "HighLow": [0,1], "Volume": []}
        margin = 1

        returns = [abs((close[i]-close[i-1])/close[i-1]) for i in range(1, len(close))]
        normalizationCoefficients["Returns"] = [0, np.max(returns)*margin]

        deltaPrice = [abs(high[i]-low[i]) for i in range(len(low))]
        normalizationCoefficients["DeltaPrice"] = [0, np.max(deltaPrice)*margin]

        normalizationCoefficients["Volume"] = [np.min(volume)/margin, np.max(volume)*margin]

        return normalizationCoefficients



    def StateProcessing(self, state, testingFlag = False):

        if testingFlag == True:
            coeffs = self.DataPreProcessing(testingFlag=True)
        else:
            coeffs = self.DataPreProcessing()

        closePrices = [state[0][i] for i in range(len(state[0]))]
        lowPrices = [state[1][i] for i in range(len(state[1]))]
        highPrices = [state[2][i] for i in range(len(state[2]))]
        volumes = [state[3][i] for i in range(len(state[3]))]


        returns = [(closePrices[i]-closePrices[i-1])/closePrices[i-1] for i in range(1, len(closePrices))]
        if coeffs["Returns"][0] != coeffs["Returns"][1]:
            state[0] = [((x - coeffs["Returns"][0])/(coeffs["Returns"][1] - coeffs["Returns"][0])) for x in returns]
        else:
            state[0] = [0 for x in returns]

        deltaPrice = [abs(highPrices[i]-lowPrices[i]) for i in range(1, len(lowPrices))]
        if coeffs["DeltaPrice"][0] != coeffs["DeltaPrice"][1]:
            state[1] = [((x - coeffs["DeltaPrice"][0])/(coeffs["DeltaPrice"][1] - coeffs["DeltaPrice"][0])) for x in deltaPrice]
        else:
            state[1] = [0 for x in deltaPrice]

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

        volumes = [volumes[i] for i in range(1, len(volumes))]
        if coeffs["Volume"][0] != coeffs["Volume"][1]:
            state[3] = [((x - coeffs["Volume"][0])/(coeffs["Volume"][1] - coeffs["Volume"][0])) for x in volumes]
        else:
            state[3] = [0 for x in volumes]

       
        # Process the state structure to obtain the appropriate format
        state = [item for sublist in state for item in sublist]

        return state

    def RewardProcessing(self, reward):
        return np.clip(reward, -self.tdqnSettings.rewardClipping, self.tdqnSettings.rewardClipping)



    def ChooseAction(self, state, previousAction, trainingFlag = True):
        
        if trainingFlag == True:
            sample = random.random()
            iteration = self.iteration
            self.iteration += 1

            if sample > self.GetEpsilon(iteration):
               return self.ChooseAction(state, previousAction, trainingFlag=False)
            return random.randrange(self.networkSettings.outputLayerSize), 0, [0, 0]

        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
            output = self.PolicyNetwork(state).squeeze(0)
            QMax, index = output.max(0)
            action = index.item()
            Q = QMax.item()
            QValues = output.cpu().numpy()
            return action, Q, QValues



    def ChooseAction2(self, state, previousAction, trainingFlag = True):
        
        if trainingFlag == True:
            # sample = random.random()
            # alphaRandom = random.random()
            iteration = self.iteration
            self.iteration += 1

            if random.random() > self.GetEpsilon(iteration):
                if random.random() > self.tdqnSettings.alpha:
                    return self.ChooseAction(state, previousAction, trainingFlag=False)
                else:
                    return previousAction, 0, [0, 0]
            else:
                return random.randrange(self.networkSettings.outputLayerSize), 0, [0, 0]


        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
            output = self.PolicyNetwork(state).squeeze(0)
            QMax, index = output.max(0)
            action = index.item()
            Q = QMax.item()
            QValues = output.cpu().numpy()
            return action, Q, QValues



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

        currentQValues = self.PolicyNetwork(state).gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            nextActions = torch.max(self.PolicyNetwork(nextState), 1).indices
            nextQValues = self.TargetNetwork(nextState).gather(1, nextActions.unsqueeze(1)).squeeze(1)
            expectedQValues = reward + self.tdqnSettings.gamma * nextQValues * (1 - done)

        self.currentLoss = F.smooth_l1_loss(currentQValues, expectedQValues)
        self.optimizer.zero_grad()
        self.currentLoss.backward()

        torch.nn.utils.clip_grad_norm_(self.PolicyNetwork.parameters(), self.tdqnSettings.gradientClipping)
        self.optimizer.step()
        
        if(self.iteration % self.tdqnSettings.targetNetworkUpdate == 0):
            self.TargetNetwork.load_state_dict(self.PolicyNetwork.state_dict())
        
        self.PolicyNetwork.eval()



    def Training(self, verbose = False):

        env = self.TrainingEnvironment
        DataAugmentation = da.DataAugmentation()
        env = DataAugmentation.generate(env)[0]
        self.currentLoss = torch.tensor(np.nan, device=self.device)

        for episode in range(self.tdqnSettings.numberOfEpisodes):

            if verbose == True:
                stockCodePadded = self.TrainingEnvironment.Position.stock.stockCode.ljust(8)
                print("{} Training: Episode {}".format(stockCodePadded, episode))

            env.reset()
            env.SetCustomStartingPoint(random.randrange(env.dataFrameLength))
            state = self.StateProcessing(env.state)
            previousAction = 0
            done = 0

            while done == 0:
                action = self.ChooseAction(state, previousAction)[0]

                nextState, reward, done = env.step(action)

                nextState = self.StateProcessing(nextState)
                reward = self.RewardProcessing(reward)
                self.ReplayMemory.Push(state, action, reward, nextState, done)

                self.LearnFromMemory()

                state = nextState
                previousAction = action
            

            self.loss.append(self.currentLoss.cpu().detach().numpy())
        
        return self.TrainingEnvironment



    def Testing(self):
        
        DataAugmentation = da.DataAugmentation()
        self.TestingEnvironment.reset()
        env = DataAugmentation.lowPassFilter(self.TestingEnvironment, 5)

        state = self.StateProcessing(env.state)
        previousAction = None
        done = 0

        while done == 0:
            action, _, _ = self.ChooseAction(state, previousAction, trainingFlag=False)
            nextState, _, done = env.step(action)
            self.TestingEnvironment.step(action)
            state = self.StateProcessing(nextState)

        return self.TestingEnvironment



    def SaveModel(self, fileName):
        lossFileName = fileName + "-loss"
        lossFile = open(lossFileName, "wb")
        pickle.dump(self.loss, lossFile)
        lossFile.close()

        modelFileName = fileName + "-model"
        torch.save(self.PolicyNetwork.state_dict(), modelFileName)



    def LoadModel(self, fileName):
        lossFileName = fileName + "-loss"
        lossFile = open(lossFileName, "rb")
        self.loss = pickle.load(lossFile)
        lossFile.close()

        modelFileName = fileName + "-model"
        model = torch.load(modelFileName, map_location=self.device)
        self.PolicyNetwork.load_state_dict(model)
        self.TargetNetwork.load_state_dict(model)



    def PlotLoss(self, saveFileName, showFlag = False):
        fig, ax1 = plt.subplots()
        ax1.plot(range(len(self.loss)), self.loss, label="Loss")
        ax1.set_xlabel("Loss", fontsize=20)
        ax1.set_ylabel("# of Episodes", fontsize=20)
        ax1.legend(loc="upper left", fontsize=14)
        ax1.tick_params(labelsize=14)
        ax1.set_yscale("log")
        # ax1.set_aspect()
        figure = plt.gcf()
        figure.set_size_inches(16,9)
        plt.tight_layout()

        plt.savefig(saveFileName + ".png", format = "png", dpi=300)
        if showFlag == True:
            plt.show()


