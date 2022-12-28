import tradingEnv as te
from dataAugmentation import DataAugmentation
import TDQN as tdqn
import torch
import random

device = torch.device('cuda:'+str(0) if torch.cuda.is_available() else 'cpu')


startingDate = '2020-01-01'
endingDate = '2022-01-01'

myte = te.TradingEnv("AAPL", startingDate=startingDate, endingDate=endingDate, money=100_000, startingPoint=1)

print(myte.data.head())
myte.setStartingPoint(50)

random.seed(10)

# for i in range(50):
#     myte.step(1)
print(myte.data[50-myte.stateLength:50])
# print(len(myte.state))

closePrices = [myte.state[0][i] for i in range(len(myte.state[0]))]
lowPrices = [myte.state[1][i] for i in range(len(myte.state[1]))]
highPrices = [myte.state[2][i] for i in range(len(myte.state[2]))]
volumes = [myte.state[3][i] for i in range(len(myte.state[3]))]

state = [item for sublist in myte.state for item in sublist]
print(state)
tensorState = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)

print(tensorState)

print(closePrices)


# dataAugmentation = DataAugmentation()
# trainingEnvList = dataAugmentation.generate(myte)
