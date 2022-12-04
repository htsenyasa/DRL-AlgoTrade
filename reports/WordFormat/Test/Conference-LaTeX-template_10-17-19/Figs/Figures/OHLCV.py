import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt 

data = yf.download("ISCTR.IS")[-5:]

mpf.plot(data, type="candle", volume=True, savefig='OHLCV.png')

