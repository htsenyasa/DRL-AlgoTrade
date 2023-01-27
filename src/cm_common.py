import pandas as pd
from itertools import zip_longest

def ReadFromFile(stockName, start, end, interval, progress):
    data = pd.read_csv("./Data/" + stockName, index_col=0)
    data = data.loc[start:end]
    data.index = pd.to_datetime(data.index)
    return data


def Grouper(iterable, n, fillValue=None): #https://stackoverflow.com/a/434411
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillValue)