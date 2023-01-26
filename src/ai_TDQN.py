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
        self.currentLoss = torch.tensor(np.nan, device=self.device)

        for episode in range(self.tdqnSettings.numberOfEpisodes):

            if verbose == True:
                stockCodePadded = self.TrainingEnvironment.Position.stock.stockCode.ljust(8)
                print("{} Training: Episode {}".format(stockCodePadded, episode))

            env.reset()
            env.SetRandomStartingPoint()
            state = env.state
            previousAction = 0
            done = 0

            while done == 0:
                action, _, _ = self.ChooseAction(state, previousAction)

                nextState, reward, done = env.step(action)

                reward = self.RewardProcessing(reward)
                self.ReplayMemory.Push(state, action, reward, nextState, done)

                self.LearnFromMemory()

                state = nextState
                previousAction = action
            
            self.loss.append(self.currentLoss.cpu().detach().numpy())
        
        return self.TrainingEnvironment



    def Testing(self):
        
        env = self.TestingEnvironment
        env.reset()
        
        state = env.state
        previousAction = None
        done = 0

        while done == 0:
            action, _, _ = self.ChooseAction(state, previousAction, trainingFlag=False)
            nextState, _, done = env.step(action)
            state = nextState

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


