import gym

class TradingEnv(gym.Env):
    def __init__(self, Position,  stateLength = 30):
        self.dataFrame = Position.stock.dataFrame.copy()
        self.stateLength = stateLength
        self.tick = stateLength
        self.horizon = Position.stock.horizon

        self.dataFrame["Position"] = Position.noPosition
        self.dataFrame["Action"] = 0
        self.dataFrame["Holdings"] = 0
        self.dataFrame["Cash"] = Position.cash
        self.dataFrame["Value"] = 0
        self.dataFrame["Returns"] = 0

        self.state = [self.dataFrame['Close'][0:stateLength].tolist(),
                      self.dataFrame['Low'][0:stateLength].tolist(),
                      self.dataFrame['High'][0:stateLength].tolist(),
                      self.dataFrame['Volume'][0:stateLength].tolist(),
                      [0]]
        self.reward = 0.
        self.done = 0

