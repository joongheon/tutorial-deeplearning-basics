from keras.models import Sequential
from keras.layers import LSTM
from sklearn.model_selection import train_test_split # pip install -U scikit-learn
import numpy as np
import matplotlib.pyplot as plt

x_data = [[[(i+j)/100] for i in range (5)] for j in range(100)]
y_data = [(i+5)/100 for i in range (100)]
x_data = np.array(x_data, dtype=float)
y_data = np.array(y_data, dtype=float)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
model=Sequential()
model.add(LSTM(1, input_dim=1, input_length=5, return_sequences = False))
model.compile(loss='mse', optimizer='adam')
model.summary()
history = model.fit(x_train, y_train, epochs=1000, verbose=0)
y_predict = model.predict(x_test)
plt.scatter(range(20), y_predict, c='r')
plt.scatter(range(20), y_test, c='g')
plt.show()
plt.plot(history.history['loss'])
plt.show()

 
