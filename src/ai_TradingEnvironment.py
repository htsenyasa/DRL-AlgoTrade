import gym
import numpy as np
from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter


Horizon = namedtuple("Horizon", ["start", "end", "interval"])

class StateObject():
    def __init__(self, principalPosition, ancillaryStocks, stateLength):
        self.Position = principalPosition
        self.ancillaryStocks = ancillaryStocks
        self.high = np.empty(stateLength, dtype=float)
        self.low = np.empty(stateLength, dtype=float)
        self.close = np.empty(stateLength, dtype=float)
        self.volume = np.empty(stateLength, dtype=float)
        self.position = False
        self.state = np.column_stack((self.high, self.low, self.close, self.volume, self.position))

    def InitScalar(self):
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.Position.dataFrame[["High", "Low", "Close", "Volume"]].values)

    def ScaleState(self, state):
        self.scaler.transform(state.T[:-1].T)


class TradingEnvironment(gym.Env):    
    def __init__(self, Position,  stateLength = 30):
        self.Position = Position 
        self.stateLength = int(stateLength)
        self.t = int(stateLength)
        self.horizon = Position.stock.horizon
        self.actions = {"LONG": 1, "SHORT": 0}

        self.dataFrame = self.Position.dataFrame
        self.dataFrameLength = len(self.dataFrame.index)
        self.State = StateObject(Position, Position, stateLength) 
        self.UpdateState()
        self.reward = 0.
        self.done = 0


    def UpdateState(self):
        currentStateRange = slice(self.t - self.stateLength, self.t)
        if self.t == self.stateLength: # For __init__ (this if may be redundant.)
            position = self.Position.NO_POSITION
        else:
            position = self.Position.position[self.t]

        # No need to copy at this stage. 
        self.State.high = self.Position.high[currentStateRange]
        self.State.low = self.Position.low[currentStateRange]
        self.State.close = self.Position.close[currentStateRange]
        self.State.volume = self.Position.volume[currentStateRange]
        self.State.position = position

        return self.State


    def InitScaler(self):
        self.Scalers = []
        columns = ["High", "Low", "Close", "Volume"]
        for i in range(columns): # No need to scale position
            Scaler = MinMaxScaler()
            Scaler.fit(self.Position.dataFrame[[columns[i]]])
            self.Scalers.append(Scaler)


    def Scale(self):
        ...


    def reset(self):
        self.Position.ResetPosition()
        
        self.reward = 0.
        self.done = 0
        self.t = self.stateLength
        
        self.dataFrame = self.Position.dataFrame
        self.UpdateState()

        return self.State


    def step(self, action):

        if action == self.actions["LONG"]:
            self.Position.GoLong()
        elif action == self.actions["SHORT"]:
            self.Position.GoShort()
            
        self.reward = self.GetReward()

        self.t += 1
        self.UpdateState()

        if self.t == self.dataFrameLength:
            self.done = 1
        
        return self.state, self.reward, self.done   


    def GetReward(self):
        return self.Position.returns[self.t]


    def SetCustomStartingPoint(self, startingPoint):
        self.t = np.clip(startingPoint, self.stateLength, len(self.dataFrame.index))
        self.UpdateState()