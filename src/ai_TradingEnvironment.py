import gym

class TradingEnvironment(gym.Env):
    
    def __init__(self, Position,  stateLength = 30):
        self.Position = Position 
        self.stateLength = stateLength
        self.tick = stateLength
        self.horizon = Position.stock.horizon

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

    def step(self):
        ...
