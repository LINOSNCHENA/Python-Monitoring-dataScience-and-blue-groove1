import numpy as np
input = np.linspace(-55, 0, 55)
input2 = np.linspace(-3, 53, 93)

def sigmoid(x):
    return 1/(1+np.exp(-x))

from matplotlib import pyplot as plt
plt.plot(input, sigmoid(input), c="r")
plt.plot(input2, sigmoid(input2), c="g")

plt.ylabel('HOUSE TEMPERATURE READINGS')
plt.show()                      # THIRD PRINT
