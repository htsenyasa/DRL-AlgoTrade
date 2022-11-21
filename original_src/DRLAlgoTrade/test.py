import tradingEnv as te

startingDate = '2020-01-01'
endingDate = '2022-01-01'

myte = te.TradingEnv("AAPL", startingDate=startingDate, endingDate=endingDate, money=100_000, startingPoint=1)

# print(myte.data.head())
myte.step(1)
myte.step(1)
myte.step(0)
myte.step(0)
myte.step(0)
myte.step(1)

print(myte.data[29:40])
