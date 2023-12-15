import numpy as np
import pandas as pd

from elbrus-ai.layers.Dense import Dense
from elbrus-ai.models.Sequential import Sequential

lst = []
lst.append([1, 2, 1])
lst.append([3, 4, 1])
lst.append([5, 6, 1])
lst.append([7, 8, 1])
print(lst)

df_train = pd.DataFrame(lst, columns=['x1', 'x2', 'y'])
X_train = df_train[['x1', 'x2']]
Y_train = df_train['y']
print(X_train)
print(Y_train)

model = Sequential()
model.add(Dense(3, 2, activation="relu"))
model.add(Dense(2, 3, activation="relu"))
model.add(Dense(1, 2, activation="sigmoid"))
model.compile(optimizer='adam', loss='MSE', metrics='accuracy')
model.fit(X_train, Y_train, epochs=1)
print('finish')