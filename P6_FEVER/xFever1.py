# Senior citizen artifical neural network
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# load the dataset
dataset = loadtxt('feverX.csv', delimiter=',')
dataset2 = loadtxt('feverY.csv', delimiter=',')
#dataset = loadtxt('smart.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
X2 = dataset2[:,0:8]
y2 = dataset2[:,8]
# define the keras model
muntu = Sequential()
muntu.add(Dense(12, input_dim=8, activation='sigmoid'))
muntu.add(Dense(8, activation='sigmoid'))
muntu.add(Dense(1, activation='sigmoid'))
# compile the keras model
muntu.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mse','acc'])
# fit the keras model on the dataset
muntu.fit(X, y, epochs=150, batch_size=10)
# evaluate the keras model
#_, accuracy = model.evaluate(X, y, verbose=0)
accuracy = muntu.evaluate(X, y)
print("loss , mse, accuracy",accuracy)
#print('Presence of moderate of fever: %.2f' % (accuracy*100))

print("=================|| SENIOR CITIZEN SUBJECT || ========================")
print("1 means Diseased, 0 means Health")
#SENIOR CITIZEN
# make class four predictions with the muntu
predictions = muntu.predict_classes(X)
for i in range(4):
	print('%s => %d (Chances of Fever Attack %d)' % (X2[i].tolist(), predictions[i], y2[i]))
print("=================|| SENIOR CITIZEN SUBJECT || =========================")

# plot metrics from trained model
from matplotlib import pyplot
history = muntu.fit(X, y,validation_split=0.33, epochs=150, batch_size=len(X), verbose=2)
print("Loss-Fever, MSE-FEVER, Accuracy-FEVER",accuracy)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
pyplot.plot(history.history['acc'])
pyplot.plot(history.history['val_acc'])
pyplot.title('model accuracy')
pyplot.ylabel('acc')
pyplot.xlabel('epoch')
pyplot.tight_layout()
pyplot.legend(['train', 'test'], loc='upper left')
pyplot.show()
# summarize history for loss
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.tight_layout()
pyplot.legend(['train', 'test'], loc='upper left')
pyplot.show()

print("Loss-Fever, MSE-FEVER, Accuracy-FEVER",accuracy)