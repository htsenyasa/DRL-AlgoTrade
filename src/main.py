import yfinance as yf
import tf_TradingOperations as to
import ai_TradingEnvironment as te
import ai_TDQN as tdqn
import ai_Network as network

networkSettings_ = network.networkSettings(inputLayerSize=100, hiddenLayerSize=100, dropout=0.2)

tdqnSettings_ = tdqn.tdqnSettings(gamma=0.4, epsilonStart=1.0, epsilonEnd=0.01,
                                epsilonDecay=10000, capacity=100000, learningRate=0.0001,
                                targetUpdateFrequency=1000, batch_size=32)

import pandas as pd
pd.options.mode.chained_assignment = None 


from collections import namedtuple

Horizon = namedtuple("Horizon", ["start", "end", "interval"])

isctrHorizon = Horizon("2020-01-01", "2022-01-01", "1d")

aapl = to.StockHandler("AAPL", yf.download, yf.download, isctrHorizon)


pos = to.DummyPosition(aapl)
pos.dataFrame["Close"] = pos.dataFrame["Adj Close"]

myte = te.TradingEnvironment(pos)
myte.step(1)
myte.step(1)
myte.step(0)
myte.step(0)
myte.step(0)
myte.step(1)


mem = tdqn.ReplayMemory(32)