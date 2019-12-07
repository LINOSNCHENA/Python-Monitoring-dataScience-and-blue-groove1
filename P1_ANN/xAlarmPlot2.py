import numpy as np

number_of_samples = 1200
low = -1
high = 0
s = np.random.uniform(low, high, number_of_samples)

# all values of s are within the half open interval [-1, 0) :

print(np.all(s >= -1) and np.all(s < 0))

import matplotlib.pyplot as plt
plt.hist(s)
plt.show()

s = np.random.binomial(100, 0.5, 1200)
plt.hist(s)
plt.show()