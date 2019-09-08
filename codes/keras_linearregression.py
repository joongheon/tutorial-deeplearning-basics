import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# Data
x_data = np.array([[1], [2], [3]])
y_data = np.array([[1], [2], [3]])
# Model, Cost, Train
model = Sequential()
model.add(Dense(1, input_dim=1))
model.compile(loss='mse', optimizer='adam')
model.fit(x_data, y_data, epochs=1000, verbose=0)
model.summary()
# Inference
print(model.get_weights())
print(model.predict(np.array([4])))
# Plot
plt.scatter(x_data, y_data)
plt.plot(x_data, y_data)
plt.grid(True)
plt.show()

