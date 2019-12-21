# NEURAL NETWORK TO DETEECK A THIEVES AT THE AUTOMATED HOUSE (PART I)

import numpy as np
feature_set = np.array([[0,1,1],[1,0,1],[1,0,0],[1,1,0],[1,0,0]])
labels = np.array([[1,0,0,1,1]])
labels = labels.reshape(5,1)
np.random.seed(84)
weights = np.random.rand(3,1)
bias = np.random.rand(1)
learningRate  = 0.1                                   # One
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoidxd(x):
    return sigmoid(x)*(1-sigmoid(x))
print('=================================== MODEL TRAINING =============================')
for epoch in range(3):
    visitorX = feature_set
    # feedforward step  1: Make a wild Guese
    XW = np.dot(feature_set, weights) + bias
    # feedforward step  2: First optimization
    z = sigmoid(XW)
    print(z.sum()/4)
    # backpropagation step 1: Find diviation
    error = z - labels
    print(error.sum())
    # backpropagation step 2: Reduce deviation/Second optimzation 
    visitorX = feature_set.T                            # Two
    z_delta = error * sigmoidxd(z)                      # Three
    weights -= learningRate * np.dot(visitorX, z_delta) # Three
    for XYZ in z_delta:
       bias -= learningRate * XYZ
print('================================== MODEL APPLICATION ==========================')
single_point = np.array([0,0,0])
result = sigmoid(np.dot(single_point, weights) + bias)
print('Chances Dangerous 1-3 # {:Bad}:',(result))
def predictx(result):
  if(result <= 0.25):
    intruder_predicted = 0
  if(result <= 0.50):
    intruder_predicted = 1
  if(result <= 0.75):
    intruder_predicted = 2
  else:
    intruder_predicted = 3
  return intruder_predicted
print((predictx(result)))

print('================================== ANN x2 END ========================')
