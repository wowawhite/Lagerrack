import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

def mycoolfunction(parameter=None):
    tellmesomething = sys.__name__
    return tellmesomething

booya = mycoolfunction()
print(booya)


def create_sequences(X, y, time_steps=4):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    # TODO: apply hamming window here,
    # TODO: apply fft for sequence here
    return np.array(Xs), np.array(ys)


arrx = np.linspace(0,100,99)
arry = np.arange(1,100)
dframe = pd.DataFrame({'date': pd.Series(arrx ),
                       'close': pd.Series(arry)})

#print(dframe)

mysequence = create_sequences(dframe["date"], dframe["close"], 20)

print((mysequence))
for a in mysequence:
    print(len(a))
