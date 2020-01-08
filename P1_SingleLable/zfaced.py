# LIBRALIES AND PACKAGES
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

police = Sequential()
# Step 1 - Convolution
police.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Step 2 - Pooling
police.add(MaxPooling2D(pool_size = (2, 2)))
police.add(Conv2D(32, (3, 3), activation = 'relu'))
police.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
police.add(Flatten())
# Step 4 - Full connection
police.add(Dense(units = 128, activation = 'relu'))
police.add(Dense(units = 1, activation = 'sigmoid'))
# Compiling the CNN
police.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - FITTING THE CNN TO THE JPG IMAGES-TYPES
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('data/training_set',
target_size = (64, 64),batch_size = 18, class_mode = 'binary')
test_set = test_datagen.flow_from_directory('data/test_set',
target_size = (64, 64),batch_size = 18,class_mode = 'binary')
police.fit_generator(training_set,steps_per_epoch = 6, epochs = 2,
validation_data = test_set, validation_steps = 20)

# Part 3 - MAKING NEW PREDICTIONS
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('data/prediction_result/unza1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = police.predict(test_image)
training_set.class_indices
if result[0][0] == 1: prediction = 'unza'
else: prediction = 'mdh'
print("=================== PREDITICTION =====================")
print (prediction)
print("====================== END =====================")