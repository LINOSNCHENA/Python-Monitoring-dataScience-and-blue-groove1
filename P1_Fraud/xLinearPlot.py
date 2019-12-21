import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import matplotlib.pyplot as plt 
 
x = data = np.linspace(1,2,20)
y = x*4 + np.random.randn(*x.shape) * 0.3

muntu = Sequential()
muntu.add(Dense(1, input_dim=1, activation='linear'))
muntu.compile(optimizer='sgd', loss='mse', metrics=['mse'])

weights = muntu.layers[0].get_weights()
w_init = weights[0][0][0]
b_init = weights[1][0]
print('Linear regression model is initialized with weights w: %.2f, b: %.2f' % (w_init, b_init)) 
muntu.fit(x,y, batch_size=1, epochs=3, shuffle=False)

weights = muntu.layers[0].get_weights()
w_final = weights[0][0][0]
b_final = weights[1][0]
print('Linear regression model is trained to have weight w: %.2f, b: %.2f' % (w_final, b_final))

predict = muntu.predict(data)
plt.plot(data, predict, 'b', data , y, 'k.')
plt.show()