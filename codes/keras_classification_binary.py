import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Data
x_data = np.array([[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]])
y_data = np.array([[0], [0], [0], [1], [1], [1]])
# Model, Cost, Train
model = Sequential()
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(x_data, y_data, epochs=10000, verbose=1)
model.summary()
# Inference
print(model.get_weights())
print(model.predict(x_data))


