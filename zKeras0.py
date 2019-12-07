# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

eHouse="file:///Users/linos/Downloads/PyData/pima-indians-diabetes.csv"
# load the dataset
dataset = loadtxt(eHouse, delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]