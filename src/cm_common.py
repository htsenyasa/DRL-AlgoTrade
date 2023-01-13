import pandas as pd

def ReadFromFile(stockName, start, end, interval, progress):
    data = pd.read_csv("./Data/" + stockName, index_col=0)
    data = data.loc[start:end]
    data.index = pd.to_datetime(data.index)
    return data