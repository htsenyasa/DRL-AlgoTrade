import gym
import numpy as np
from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter
import random


Horizon = namedtuple("Horizon", ["start", "end", "interval"])

class StateObject():
    def __init__(self, principalPosition, ancillaryStocks):
        self.Position = principalPosition # Must be a reference to the position class passed to the trading env.
        self.ancillaryStocks = ancillaryStocks
        self.__columns = ["Open", "High", "Low", "Close", "Volume"]
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.Position.dataFrame[self.__columns].values) # Default Scaling
        # self.__InitIndexScaler()

    def InitScaler(self, dataFrame):
        self.scaler = MinMaxScaler()
        self.scaler.fit(dataFrame[self.__columns].values)

    def ScaleState(self, partialState):
        return self.scaler.transform(partialState)

    def __InitIndexScaler(self):
        scaler = MinMaxScaler()
        self.bistClose = scaler.fit_transform(self.ancillaryStocks.dataFrame["Close"].values.reshape(-1,1)).flatten()

    def GetState(self, currentRange, position):
        partialState = self.ScaleState(np.column_stack((self.Position.open[currentRange],
                                                        self.Position.high[currentRange],
                                                        self.Position.low[currentRange],
                                                        self.Position.close[currentRange],
                                                        self.Position.volume[currentRange])))
        # partialState = np.column_stack((partialState, self.bistClose[currentRange]))
        # return np.concatenate((partialState.flatten("F"), [position]))
        return np.concatenate((partialState.flatten("F"), [position])).tolist()
        

class TradingEnvironment(gym.Env):    
    def __init__(self, Position, ancillaryStocks = None, stateLength = 30):
        self.Position = Position

        self.stateLength = int(stateLength)
        self.SetStartingPoint(self.stateLength)

        self.horizon = Position.stock.horizon
        self.actions = {"LONG": 1, "SHORT": 0}

        self.__State = StateObject(Position, ancillaryStocks)
        self.state = self.__UpdateState()
        self.reward = 0.
        self.done = 0


    def InitScaler(self, dataFrame):
        self.__State.InitScaler(dataFrame)


    def __UpdateState(self):
        currentRange = slice(self.t - self.stateLength, self.t)
        if self.t == self.stateLength: # For __init__ (this may be redundant.)
            position = self.Position.NO_POSITION
        else:
            position = self.Position.position[self.t]

        return self.__State.GetState(currentRange, position)


    def reset(self):
        self.Position.ResetPosition()
        
        self.reward = 0.
        self.done = 0
        self.t = self.stateLength
        
        self.state = self.__UpdateState()

        return self.state



    def step(self, action):

        if action == self.actions["LONG"]:
            self.Position.GoLong()
        elif action == self.actions["SHORT"]:
            self.Position.GoShort()
            
        self.reward = self.GetReward()

        self.state = self.__UpdateState()
        self.t += 1

        self.done = self.CheckDoneSignal()
        
        return self.state, self.reward, self.done   



    def CheckDoneSignal(self):
        # if self.t == self.Position.Length or self.Position.value[self.t-1] < self.Position.initialCash * 0.6:
        if self.t == self.Position.Length:
            return 1
        return 0



    def GetReward(self):
        return self.Position.returns[self.t]



    def SetStartingPoint(self, startingPoint):
        self.t = np.clip(startingPoint, self.stateLength, self.Position.Length)
        self.Position.SetStartingPoint(self.t)



    def SetRandomStartingPoint(self, randRange = None):
        if randRange == None:
            self.SetStartingPoint(random.randrange(1, self.Position.Length))
        else:
            self.SetStartingPoint(random.randrange(*randRange))



    def NewActionBranch(self, depth=1):
        self.__tempT = self.t
        self.__tempState = self.state
        self.__tempDone = self.done
        self.Position.NewActionBranch(depth)


    
    def MergeBranches(self):
        self.t = self.__tempT
        self.state = self.__tempState
        self.done = self.__tempDone
        self.Position.MergeBranches()
