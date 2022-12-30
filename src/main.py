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
                                  numberOfEpisodes = 50, 
                                  rewardClipping = 1
                                  )

optimSettings = tdqn.optimSettings_(L2Factor=0.000001)


startingDate = '2012-1-1'
splittingDate = '2018-1-1'
endingDate = '2020-1-1'


trainingHorizon = te.Horizon(startingDate, splittingDate, "1d")
testingHorizon = te.Horizon(splittingDate, endingDate, "1d")

aaplStockTraining = to.StockHandler("AAPL", yf.download, yf.download, trainingHorizon)
aaplStockTesting = to.StockHandler("AAPL", yf.download, yf.download, testingHorizon)

posTraining = to.DummyPosition(aaplStockTraining)
posTesting = to.DummyPosition(aaplStockTesting)

trainingEnvironment = te.TradingEnvironment(posTraining)
testingEnvironment = te.TradingEnvironment(posTesting)

agent = tdqn.TDQNAgent(trainingEnvironment, testingEnvironment, tdqnSettings, networkSettings, optimSettings)
agent.Training()
agent.Testing()
buyHistory, sellHistory = posTesting.ParseActions()

index = -100
apd = [mpf.make_addplot(buyHistory[index:],type='scatter', markersize=50,marker='^'), mpf.make_addplot(sellHistory[index:],type='scatter', markersize=50,marker='v')]
mpf.plot(posTesting.dataFrame[index:], addplot=apd, type="candle")

# for s in state:
#     print("{:.5f}".format(s))

# print(coeffs)


