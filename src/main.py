import yfinance as yf
import tf_TradingOperations as to
import ai_TradingEnvironment as te
import ai_TDQN as tdqn
import ai_Network as network
import pandas as pd
pd.options.mode.chained_assignment = None
from collections import namedtuple
import torch
import random

device = torch.device('cuda:'+str(0) if torch.cuda.is_available() else 'cpu')


networkSettings = tdqn.networkSettings_(inputLayerSize=100, hiddenLayerSize=100, outputLayerSize=2, dropout=0.2)

tdqnSettings = tdqn.tdqnSettings_(gamma=0.4, epsilonStart=1.0, epsilonEnd=0.01,
                                epsilonDecay=10000, capacity=100000, learningRate=0.0001,
                                targetUpdateFrequency=1000, batchSize=32, gradientClipping=1, targetNetworkUpdate=1000)

optimSettings = tdqn.optimSettings_(L2Factor=0.000001)


Horizon = namedtuple("Horizon", ["start", "end", "interval"])
startingDate = '2012-1-1'
endingDate = '2020-1-1'
splittingDate = '2018-1-1'
isctrHorizon = Horizon(startingDate, endingDate, "1d")
aapl = to.StockHandler("AAPL", yf.download, yf.download, isctrHorizon)
pos = to.DummyPosition(aapl)
pos.dataFrame["Close"] = pos.dataFrame["Adj Close"]

myte = te.TradingEnvironment(pos)



mem = tdqn.ReplayMemory(100000)
random.seed(10)
for i in range(500):
    action = random.randint(0,1)
    state, reward, done, info = myte.step(action)
    mem.Push(state, action, reward, state, done)

# print(myte.dataFrame[525:535][["Cash", "Action", "Holdings"]])


agent = tdqn.TDQNAgent(myte, tdqnSettings, networkSettings, optimSettings)
# coeffs = agent.DataPreProcessing()
myte.SetCustomStartingPoint(60)
state = agent.StateProcessing(myte.state)
# print(state, sep='\n')

for s in state:
    print("{:.5f}".format(s))

# print(coeffs)