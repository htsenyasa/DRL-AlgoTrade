import gym
from collections import namedtuple

class TradingEnvironment(gym.Env):
    
    def __init__(self, Position,  stateLength = 30):
        self.Position = Position 
        self.stateLength = stateLength
        self.tick = stateLength
        self.horizon = Position.stock.horizon
        self.actions = {"LONG": 1, "SHORT": 0}

        self.dataFrame = self.Position.dataFrame
        self.state = [self.dataFrame['Close'][0:self.stateLength].tolist(),
                      self.dataFrame['Low'][0:self.stateLength].tolist(),
                      self.dataFrame['High'][0:self.stateLength].tolist(),
                      self.dataFrame['Volume'][0:self.stateLength].tolist(),
                      [0]]
        self.reward = 0.
        self.done = 0

    def reset(self):
        self.Position.ResetPosition()

        self.dataFrame = self.Position.dataFrame
        self.state = [self.dataFrame['Close'][0:self.stateLength].tolist(),
                      self.dataFrame['Low'][0:self.stateLength].tolist(),
                      self.dataFrame['High'][0:self.stateLength].tolist(),
                      self.dataFrame['Volume'][0:self.stateLength].tolist(),
                      [0]]
        self.reward = 0.
        self.done = 0
        self.tick = self.stateLength

        return self.state

    def step(self, action):

        if action == self.actions["LONG"]:
            self.Position.GoLong(self.tick)
        elif action == self.actions["SHORT"]:
            self.Position.GoShort(self.tick)
            
        if (self.Position.IsShort(self.tick)  and  self.Position.IsShort(self.tick-1)):
            self.reward = (self.dataFrame["Close"][self.tick-1] - self.dataFrame["Close"][self.tick])/self.dataFrame["Close"][self.tick-1]
        else:
            self.reward = self.dataFrame["Returns"][self.tick]

        self.tick += 1
        self.state = [self.dataFrame['Close'][(self.tick - self.stateLength) : self.tick].tolist(),
                      self.dataFrame['Low'][(self.tick - self.stateLength) : self.tick].tolist(),
                      self.dataFrame['High'][(self.tick - self.stateLength) : self.tick].tolist(),
                      self.dataFrame['Volume'][(self.tick - self.stateLength) : self.tick].tolist(),
                      self.dataFrame["Position"][self.tick-1]]
        
        if(self.tick == self.dataFrame.shape[0]):
            self.done = 1  
