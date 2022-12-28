import yfinance as yf
import tf_TradingOperations as to
import ai_TradingEnvironment as te
import ai_TDQN as tdqn
import ai_Network as network
import torch

networkSettings = tdqn.networkSettings_(inputLayerSize=100, hiddenLayerSize=100, outputLayerSize=2, dropout=0.2)

tdqnSettings = tdqn.tdqnSettings_(gamma=0.4, epsilonStart=1.0, epsilonEnd=0.01,
                                epsilonDecay=10000, capacity=100000, learningRate=0.0001,
                                targetUpdateFrequency=1000, batchSize=32, gradientClipping=1, targetNetworkUpdate=1000)

optimSettings = tdqn.optimSettings_(L2Factor=0.000001)

horizon = to.Horizon("2018-01-01", "2022-01-01", "1d")
trainingHorizon = to.Horizon("2018-01-01", "2020-12-31", "1d")
testingHorizon = to.Horizon("2021-01-01", "2022-01-01", "1d")

agent = tdqn.TDQNAgent(tdqnSettings, networkSettings, optimSettings)

print(agent.GetEpsilon())

mem = tdqn.ReplayMemory(32)


mem.Push([1,2,3], 1, 0.3, [2,3,4], 0)
mem.Push([132,7,33], 1, 0.3, [2,3,4], 0)

print(torch.max(torch.rand(4,4), 1)[1])
print(torch.max(torch.rand(4,4), 1).indices)

