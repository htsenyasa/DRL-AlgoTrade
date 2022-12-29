import gym
import numpy as np

class TradingEnvironment(gym.Env):    
    def __init__(self, Position,  stateLength = 30):
        self.Position = Position 
        self.stateLength = stateLength
        self.tick = int(stateLength)
        self.horizon = Position.stock.horizon
        self.actions = {"LONG": 1, "SHORT": 0}

        self.dataFrame = self.Position.dataFrame
        self.state = self.UpdateState()
        self.reward = 0.
        self.done = 0

    def UpdateState(self):
        currentStateRange = slice(self.tick - self.stateLength, self.tick)
        if self.tick == self.stateLength: 
            position = self.Position.NO_POSITION
        else:
            position = self.dataFrame["Position"][self.tick-1]

        if self.tick == self.dataFrame.shape[0]:
            self.done = 1

        return [self.dataFrame['Close'][currentStateRange].tolist(),
                self.dataFrame['Low'][currentStateRange].tolist(),
                self.dataFrame['High'][currentStateRange].tolist(),
                self.dataFrame['Volume'][currentStateRange].tolist(),
                [position]]


    def reset(self):
        self.Position.ResetPosition()
        
        self.reward = 0.
        self.done = 0
        self.tick = self.stateLength
        
        self.dataFrame = self.Position.dataFrame
        self.state = self.UpdateState()

        return self.state


    def step(self, action):
        # Save tick and tick - 1 entries of the dataFrame to realize action branching -> Action, oppositeAction
        # This is a temporary workaround for oppositeAction.
        tempDataFramePrevTick = self.dataFrame.iloc[self.tick-1]
        tempDataFrameTick = self.dataFrame.iloc[self.tick]

        # Take the opposite action first.
        oppositeAction = int(not bool(action))
        if oppositeAction == self.actions["LONG"]:
            self.Position.GoLong(self.tick)
        elif oppositeAction == self.actions["SHORT"]:
            self.Position.GoShort(self.tick)

        self.oppositeActionState = self.UpdateState()

        oppositeActionReward = self.GetReward()
        self.oppositeActionInfo = {"State": self.oppositeActionState, "Reward": oppositeActionReward}

        # Replace by the old data back.
        self.dataFrame.iloc[self.tick-1] = tempDataFramePrevTick
        self.dataFrame.iloc[self.tick] = tempDataFrameTick

        if action == self.actions["LONG"]:
            self.Position.GoLong(self.tick)
        elif action == self.actions["SHORT"]:
            self.Position.GoShort(self.tick)
            
        self.reward = self.GetReward()

        self.tick += 1
        self.state = self.UpdateState()
        
        # if(self.tick == self.dataFrame.shape[0]):
        #     self.done = 1

        # print("REWARD: {:.3f}".format(self.reward))

        return self.state, self.reward, self.done, self.oppositeActionInfo         


    def GetReward(self):
        if (self.Position.IsShort(self.tick -1)  and  self.dataFrame["Action"][self.tick] == self.Position.SHORT):
            return (self.dataFrame["Close"][self.tick-1] - self.dataFrame["Close"][self.tick])/self.dataFrame["Close"][self.tick-1]
        return self.dataFrame["Returns"][self.tick]


    def SetCustomStartingPoint(self, startingPoint):
        self.tick = np.clip(startingPoint, self.stateLength, len(self.dataFrame.index))
        self.state = self.UpdateState()