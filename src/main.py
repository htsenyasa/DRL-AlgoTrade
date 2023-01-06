import yfinance as yf
import tf_TradingOperations as to
import ai_TradingEnvironment as te
import ai_TDQN as tdqn
import ai_Network as network
import pandas as pd
pd.options.mode.chained_assignment = None
import torch
import numpy as np
import mplfinance as mpf
import tf_DataAugmentation as da
import matplotlib.pyplot as plt
import random
import time


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
                                  targetUpdateFrequency=1000, 
                                  batchSize=32, 
                                  gradientClipping=1,
                                  targetNetworkUpdate=1000, 
                                  alpha=0.1, 
                                  numberOfEpisodes = 100, 
                                  rewardClipping = 1
                                  )

optimSettings = tdqn.optimSettings_(L2Factor=0.000001)


startingDate = '2012-1-1'
splittingDate = '2021-1-1'
endingDate = '2023-1-1'


trainingHorizon = te.Horizon(startingDate, splittingDate, "1d")
testingHorizon = te.Horizon(splittingDate, endingDate, "1d")

aaplStockTraining = to.StockHandler("AAPL", yf.download, yf.download, trainingHorizon)
aaplStockTesting = to.StockHandler("AAPL", yf.download, yf.download, testingHorizon)

posTraining = to.DummyPosition(aaplStockTraining)
posTesting = to.DummyPosition(aaplStockTesting)

trainingEnvironment = te.TradingEnvironment(posTraining)
testingEnvironment = te.TradingEnvironment(posTesting)

fileName = "APPL-2123-v1"

start = time.time()

agent = tdqn.TDQNAgent(trainingEnvironment, testingEnvironment, tdqnSettings, networkSettings, optimSettings)
agent.Training(verbose=True)
agent.SaveModel(fileName)
# agent.LoadModel("MyModel3")
agent.Testing()

end = time.time()
print("In Seconds: {}".format(end - start))
print("In Minutes: {}".format((end - start)/60))

posTesting.PlotActionsCapital(fileName + "-Capital", showFlag=True)
agent.PlotLoss(fileName + "-Loss", showFlag=True)
