import numpy as np
import pandas as pd

from layers.Dense import Dense
from models.Sequential import Sequential

lst = []
lst.append([2, 3, 1])
lst.append([2, 3, 1])
lst.append([2, 3, 1])
lst.append([2, 3, 1])
lst.append([2, 3, 1])
lst.append([2, 3, 1])
lst.append([2, 3, 1])
print(lst)

df_train = pd.DataFrame(lst, columns=['x1', 'x2', 'y'])
print(df_train)
X_train = df_train[['x1', 'x2']]
Y_train = df_train['y']

model = Sequential()
model.add(Dense(3, 2))
model.add(Dense(2, 3))
model.add(Dense(1, 2))
model.compile(optimizer='adam', loss='MSE', metrics='accuracy')
model.fit(X_train, Y_train, epochs=1)
print('finish')