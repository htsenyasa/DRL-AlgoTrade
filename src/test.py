import multiprocessing as mp
import yfinance as yf
import tf_TradingOperationsNew as to
import ai_TradingEnvironment as te
import ai_TDQN as tdqn
import pandas as pd
pd.options.mode.chained_assignment = None
import torch
import numpy as np
import random
import os
from cm_common import ReadFromFile
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter


device = torch.device('cuda:'+str(0) if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


networkSettings = tdqn.networkSettings_(inputLayerSize=117, hiddenLayerSize=512, outputLayerSize=2, dropout=0.2)

tdqnSettings = tdqn.tdqnSettings_(gamma=0.4, 
                                  epsilonStart=1.0, 
                                  epsilonEnd=0.01,
                                  epsilonDecay=10000,
                                  capacity=100000, 
                                  learningRate=0.0001,
                                  targetUpdateFrequency=500, 
                                  batchSize=64, 
                                  gradientClipping=1,
                                  targetNetworkUpdate = 1000, 
                                  alpha=0.1, 
                                  numberOfEpisodes = 100,
                                  rewardClipping = 1
                                  )

optimSettings = tdqn.optimSettings_(L2Factor=0.000001)


startingDate = '2019-01-01'
splittingDate = '2021-01-01'
endingDate = '2023-01-01'


trainingHorizon = te.Horizon(startingDate, splittingDate, "1d")

stock = to.StockHandler("AAPL", ReadFromFile, ReadFromFile, trainingHorizon)
pos = to.DummyPosition(stock)
env = te.TradingEnvironment(pos)

plt.plot(env.Position.dateTimeIndex, env.Position.close, label="Original")
plt.plot(env.Position.dateTimeIndex, gaussian_filter(env.Position.close, sigma=0.7)+10, label="Gaussian")
plt.plot(env.Position.dateTimeIndex, env.Position.dataFrame["Close"].rolling(window=4).mean()+20, label="Filter")
plt.legend()
plt.show()

# scaler = MinMaxScaler()
# scaler.fit(env.Position.close.reshape(-1,1))
# scaled = scaler.transform(env.Position.close.reshape(-1,1))


# plt.plot(env.Position.dateTimeIndex, scaled)
# plt.show()