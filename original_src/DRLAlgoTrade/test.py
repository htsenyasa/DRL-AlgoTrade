import tradingEnv as te
from dataAugmentation import DataAugmentation
import TDQN as tdqn
import torch
import random

startingDate = '2012-1-1'
endingDate = '2020-1-1'
splitingDate = '2018-1-1'

stateLength = 30
observationSpace = 1 + (stateLength-1)*4
actionSpace = 2

numberOfEpisodes = 50
trainingParameters = [numberOfEpisodes]

# startingDate = '2020-01-01'
# endingDate = '2022-01-01'

myte = te.TradingEnv("AAPL", startingDate=startingDate, endingDate=endingDate, money=100_000)

agent = tdqn.TDQN(observationSpace=observationSpace, actionSpace=actionSpace)
# agent.training(myte, trainingParameters=trainingParameters, verbose=True, rendering=False, plotTraining=False, showPerformance=False)

mem = tdqn.ReplayMemory()
random.seed(12)
for i in range(500):
    action = random.randint(0,1)
    state, reward, done, info = myte.step(action)
    mem.push(state, action, reward, state, done)

# print(myte.data[525:535][["Cash", "Action", "Holdings"]])


coeffs = agent.getNormalizationCoefficients(myte)
myte.setStartingPoint(60)
state = agent.processState(myte.state, coeffs)

action, _, _ = agent.chooseAction(state)

print(action)

# for s in state:
#     print("{:.5f}".format(s))

# print(coeffs)