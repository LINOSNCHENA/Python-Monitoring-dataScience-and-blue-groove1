# Senior citizen artifical neural network
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# load the dataset
dataset = loadtxt('zambia.csv', delimiter=',')
dataset2 = loadtxt('xpemba.csv', delimiter=',')
#dataset = loadtxt('smart.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
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
muntu.fit(X, y,validation_split=0.33, epochs=500, batch_size=10, verbose=2)
# evaluate the keras muntu
accuracy = muntu.evaluate(X, y)
print("Loss-Fever, MSE-FEVER, Accuracy-FEVER",accuracy)
print("================= RESULT OF PREDICTION FOR TWO SUBJECT PERSONS  ========================")
print("1 means Diseased, 0 means Health")
# make class four predictions with the muntu
predictions = muntu.predict_classes(X)
for i in range(2):
	print('%s => %d (Prediction of Fever Attack %d)' % (X2[i].tolist(), predictions[i], y2[i]))
print("================= RESULT OF PREDICTION FOR TWO SUBJECT PERSONS  ========================")

# plot metrics from trained model
from matplotlib import pyplot
history = muntu.fit(X, y,validation_split=0.33, epochs=500, batch_size=10, verbose=2)
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
pyplot.legend(['train', 'test'], loc='upper left')
pyplot.show()

# summarize for subject of fever test
pyplot.plot(range(2))
pyplot.plot(predictions)
pyplot.title('subjected persons results for fever testing')
pyplot.ylabel('predictions')
pyplot.xlabel('2 hours Intervals')
pyplot.legend(['trained', 'subject'], loc='upper left')
pyplot.show()

print("Loss-FEVER, MSE-FEVER, Accuracy-FEVER :::",accuracy)
print(muntu.summary())