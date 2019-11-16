# PLOT OF TEMPERATURE IN THE AUTOMATED HOUSE
 
import numpy as np
input = np.linspace(-35, 0, 35)
input2 = np.linspace(-3, 53, 93)

def sigmoid(x):
    return 1/(1+np.exp(-x))

from matplotlib import pyplot as plt
plt.plot(input, sigmoid(input), c="r")
plt.plot(input2, sigmoid(input2), c="g")

plt.ylabel('AUTOMATED HOUSE TEMPERATURE MEASURMENTS')
plt.show()                                #           THIRD PRINT
