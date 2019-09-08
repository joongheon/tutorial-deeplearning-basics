import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Data
x_data = np.array([[1,2,1,1], [2,1,3,2], [3,1,3,4], [4,1,5,5], [1,7,5,5], [1,2,5,6], [1,6,6,6], [1,7,7,7]])
y_data = np.array([[0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [1,0,0], [1,0,0]])

# Model, Cost, Train
model = Sequential()
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(x_data, y_data, epochs=10000, verbose=1)
model.summary()
# Inference
y_predict = model.predict(np.array([[1,11,7,9]]))
print(y_predict)
print("argmax: ", np.argmax(y_predict))


