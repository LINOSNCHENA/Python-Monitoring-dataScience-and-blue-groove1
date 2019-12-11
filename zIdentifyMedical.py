# Senior citizen artifical neural network
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as pyplot
import numpy

# load the dataset
dataset1 = loadtxt('zpemba1.csv', delimiter=',')
dataset2 = loadtxt('zpemba2.csv', delimiter=',')

X = dataset1[:,0:8]
y = dataset1[:,8]
X2 = dataset2[:,0:8]
y2 = dataset2[:,8]
# define the keras muntu
muntu = Sequential()
muntu.add(Dense(12, input_dim=8, activation='sigmoid'))
muntu.add(Dense(8, activation='sigmoid'))
muntu.add(Dense(1, activation='sigmoid'))
# compile the keras muntu
muntu.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mse','acc'])
# fit the keras muntu on the dataset
muntu.fit(X, y,validation_split=0.33, epochs=150, batch_size=len(X), verbose=2)
# evaluate the keras muntu
accuracy = muntu.evaluate(X, y)
print("Loss-Fever, MSE-FEVER, Accuracy-FEVER",accuracy)
print("================= RESULT OF PREDICTION FOR TWO SUBJECT PERSONS  ========================")
print("1 means Diseased, 0 means Health")
# make class four predictions with the muntu
predictions = muntu.predict_classes(X)
for i in range(2):
	print('%s => %d (Prediction of Fever Attack %d)' % (X[i].tolist(), predictions[i], y[i]))
print("================= RESULT OF PREDICTION FOR TWO SUBJECT PERSONS  ========================")

# plot metrics from trained model
history = muntu.fit(X, y,validation_split=0.33, epochs=150, batch_size=len(X), verbose=2)
print("Loss-Fever, MSE-FEVER, Accuracy-FEVER",accuracy)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
pyplot.plot(history.history['acc'])
pyplot.plot(history.history['val_acc'])
pyplot.title('Training and Testing accuracy in fever detection')
pyplot.ylabel('accuracy levels of function')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'test'], loc='upper left')
pyplot.show()
# summarize history for loss
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('Training and testing loss in fever detection')
pyplot.ylabel('loss levels of function')
pyplot.xlabel('epoch')
pyplot.tight_layout()
pyplot.legend(['train', 'test'], loc='best')
pyplot.show()
# summarize for subject of fever test
pyplot.plot(range(2))
pyplot.plot(predictions)
pyplot.title('subjected persons results for fever testing')
pyplot.ylabel('predictions')
pyplot.xlabel('2 hours Intervals')
pyplot.tight_layout()
pyplot.legend(['trained', 'subject'], loc='best')
pyplot.show()

print("Loss-FEVER, MSE-FEVER, Accuracy-FEVER :::",accuracy)
print(muntu.summary())