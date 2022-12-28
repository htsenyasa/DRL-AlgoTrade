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
isctrHorizon = Horizon("2020-01-01", "2022-01-01", "1d")
aapl = to.StockHandler("AAPL", yf.download, yf.download, isctrHorizon)
pos = to.DummyPosition(aapl)
pos.dataFrame["Close"] = pos.dataFrame["Adj Close"]

random.seed(10)


myte = te.TradingEnvironment(pos)
myte.SetCustomStartingPoint(50)

for i in range(50):
    action = random.randrange(0,1)
    myte.step(action)
print(myte.dataFrame[50-myte.stateLength:50])
# print(myte.state)

tensorState = torch.tensor(myte.state, dtype=torch.float, device=device).unsqueeze(0)


