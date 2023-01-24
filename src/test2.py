import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd



a = np.random.random(10)
b = np.random.random(10)
x = np.repeat(1, 10)

data = pd.DataFrame({"a": a, "b": b})

scaler = MinMaxScaler()
scaler.fit(a.reshape(-1,1))
print(scaler.transform(a.reshape(-1,1)))

scaler2 = MinMaxScaler()
scaler2.fit(b.reshape(-1,1))
print(scaler2.transform(b.reshape(-1,1)))


c = np.column_stack((a, b, x))
scaler3 = MinMaxScaler()
scaler3.fit(c.T[:-1].T)
c = np.column_stack((scaler3.transform(c.T[:-1].T), c.T[-1]))

# scaler4 = MinMaxScaler()
# scaler4.fit(data.values)
# print(scaler4.transform(c))