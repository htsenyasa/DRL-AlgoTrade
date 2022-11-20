import yfinance as yf
import tf_TradingOperations as to
import ai_TradingEnvironment as te

import pandas as pd
pd.options.mode.chained_assignment = None 


from collections import namedtuple

Horizon = namedtuple("Horizon", ["start", "end", "interval"])

isctrHorizon = Horizon("2020-01-01", "2022-01-01", "1d")

aapl = to.StockHandler("AAPL", yf.download, yf.download, isctrHorizon)

pos = to.DummyPosition(aapl)

teISCTR = te.TradingEnvironment(pos)

pos.GoLong(30)
pos.GoLong(31)
pos.GoShort(32)
print(pos.dataFrame[30:40])
