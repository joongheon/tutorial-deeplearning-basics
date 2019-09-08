import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Data
xy = np.loadtxt('data-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Model, Cost, Train
model = Sequential()
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(x_data, y_data, epochs=1000, verbose=1)
model.summary()


