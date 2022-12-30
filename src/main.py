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


networkSettings = tdqn.networkSettings_(inputLayerSize=117, hiddenLayerSize=512, outputLayerSize=2, dropout=0.2)

tdqnSettings = tdqn.tdqnSettings_(gamma=0.4, 
                                  epsilonStart=1.0, 
                                  epsilonEnd=0.01,
                                  epsilonDecay=10000,
                                  capacity=100000, 
                                  learningRate=0.0001,
                                  targetUpdateFrequency=1000, 
                                  batchSize=32, 
                                  gradientClipping=1,
                                  targetNetworkUpdate=1000, 
                                  alpha=0.1, 
                                  numberOfEpisodes = 5, 
                                  rewardClipping = 1
                                  )

optimSettings = tdqn.optimSettings_(L2Factor=0.000001)


Horizon = namedtuple("Horizon", ["start", "end", "interval"])
startingDate = '2019-1-1'
endingDate = '2020-1-1'
# splittingDate = '2018-1-1'
isctrHorizon = Horizon(startingDate, endingDate, "1d")
aapl = to.StockHandler("AAPL", yf.download, yf.download, isctrHorizon)
pos = to.DummyPosition(aapl)
pos.dataFrame["Close"] = pos.dataFrame["Adj Close"]

myte = te.TradingEnvironment(pos)




agent = tdqn.TDQNAgent(myte, myte, tdqnSettings, networkSettings, optimSettings)
agent.Training()



# for s in state:
#     print("{:.5f}".format(s))

# print(coeffs)