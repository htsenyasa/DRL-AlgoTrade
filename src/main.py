import yfinance as yf
import tf_TradingOperations as to

from collections import namedtuple

Horizon = namedtuple("Horizon", ["start", "end", "interval"])

isctrHorizon = Horizon("2020-01-01", "2022-01-01", "1d")

isctr = to.StockHandler("ISCTR.IS", yf.download, yf.download, isctrHorizon)