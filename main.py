import numpy as np
import pandas as pd

from elbrusAI.layers import Dense
from elbrusAI.models import Sequential


a = np.array([[1, 1], [1, 1], [1, 1]])
print(a)
b = np.array([[1], [0], [-1]])
print(b)
print(a-b)
print("=================")

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
model.add(Dense(units=3, input_shape=2, activation="sigmoid"))
model.add(Dense(units=2, input_shape=3, activation="sigmoid"))
model.add(Dense(units=1, input_shape=2, activation="sigmoid"))
model.compile(optimizer='SGD', loss='MSE', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=1)
print('finish')

# model = Sequential()
# model.add(Dense(72, activation='relu', input_shape=(X_train.shape[1],)))
# model.add(Dense(72, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam', loss='BinaryCrossentropy', metrics=['accuracy'])
#
# model.fit(X_train,
#           Y_train,
#           epochs=45,
#           batch_size=8,
#           validation_split=0.1,
#           verbose=1)
#
# Y_pred = model.predict(X_test)