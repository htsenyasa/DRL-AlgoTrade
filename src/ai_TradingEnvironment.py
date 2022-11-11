import gym

class TradingEnv(gym.Env):
    def __init__(self, Stock, Position,  stateLength = 30):