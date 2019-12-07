# NEURAL NETWORK TO DETECT A DISEASE AT THE AUTOMATED HOUSE (PART I)
import numpy as np
feature_set = np.array([[0,0,1],[1,0,1],[1,1,0],[1,0,1]])
labels = np.array([[0,1,1,0]])
labels = labels.reshape(4,1)
np.random.seed(84)
weights = np.random.rand(3,1)
bias = np.random.rand(1)
learningRate  = 0.1                                   # One
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoidxd(x):
    return sigmoid(x)*(1-sigmoid(x))
print('======================================== AAN TRAINING BODY ==========================')
for epoch in range(13):
    visitorX = feature_set
    # feedforward step  1: Make a wild Guese
    XW = np.dot(feature_set, weights) + bias
    # feedforward step  2: First optimization
    z = sigmoid(XW)
    # backpropagation step 1: Find diviation
    error = z - labels
    print("FowardValueX : "+ str(z.sum()/4)+"- : ErrorValueX : "+ str(error.sum()))
    # backpropagation step 2: Reduce deviation/Second optimzation 
    visitorX = feature_set.T                            # Two
    z_delta = error * sigmoidxd(z)                      # Three
    weights -= learningRate * np.dot(visitorX, z_delta) # ALL Combined
    for XYZ in z_delta:
       bias -= learningRate * XYZ
print('======================================== ANN RESULTS OF QUERY ===================')
patientX = np.array([0,1,0])
result = sigmoid(np.dot(patientX, weights) + bias)
print (result)
def predictx(result):
  if(result < 70/100)  :   disease_predicted = 70,  print("r is less than 70")
  elif(result < 80/100):   disease_predicted = 80,  print("r is less than 80")
  elif(result < 90/100):   disease_predicted = 90,  print("r is less than 90")
  else                 :   disease_predicted = 100, print("r is Excellent xx")
  return disease_predicted
print('0 = health, 1 = pending, 2 = sick')
print("Monitoring unit :", result,'   -|||-   Max-Health Interpretation :', (predictx(result)))
print("=================================================================================")
