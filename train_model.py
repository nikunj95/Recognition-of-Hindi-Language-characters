# In this file, training of the model is performed. Import all the necessary
# module from keras. Create a training generator and validation generator.
# Initialise the model and add the layers to the network. Compile and start
# training the model. Run the training for 10 epochs and save the model. Also
# save the history of the training into a file.
# the dataset has a set of training and validation images that need to be loaded

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
import pickle

# image size
img_width, img_height = 32, 32

# directory for training and validation image set
train_dir = 'data/Train'
validation_dir = 'data/Validation'

train_samples = 78200 # 1700x46
validation_samples = 13754 # 300x40 + 294x6

# create an image data generator for trainiig samples
train_datagen = ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
	train_dir,
	target_size=(img_width,img_height),
	batch_size=32,
	class_mode='categorical')

# create an image data generator for validation samples
test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
	validation_dir,
	target_size=(img_width,img_height),
	batch_size=32,
	class_mode='categorical')

# Define model and add all the layers for our network
model = Sequential()
model.add(Conv2D(32, 3,3, activation='relu',input_shape=(img_width,img_height, 3)))
model.add(Conv2D(64, 3,3, activation='relu'))
model.add(MaxPool2D(2,2))

model.add(Conv2D(64, 3,3, activation='relu'))
model.add(Conv2D(64, 3,3, activation='relu'))
model.add(MaxPool2D(2,2))

model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

model.add(Dense(46, activation='softmax'))
# end of model

# compile the model
model.compile(loss='categorical_crossentropy',
	optimizer='adam',
	metrics=['accuracy'])

# no of iterations
epochs = 10

# fit the model with fit generator
history = model.fit_generator(train_generator,
	steps_per_epoch=train_samples,
	nb_epoch=epochs,
	validation_data=validation_generator,
	validation_steps=validation_samples)

# save the trained model
model.save('my_test_model2.2.h5')

# save the history object which has details on the training
with open("history_file", "w") as f:
	pickle.dump(history, f, -1)

# End of training
print "Done."




# --------------------------------------- #
# plot the accuracy and loss graph

# import matplotlib.pyplot as plt
# import pickle

# with open ('history2_file', 'r') as f:
# 	obj = pickle.load(f)

# fig1, ax_acc = plt.subplots()
# plt.plot(obj.history['acc'])
# plt.plot(obj.history['val_acc'])
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Model - Accuracy')
# plt.legend(['Training', 'Validation'], loc='lower right')
# plt.show()

# fig2, ax_loss = plt.subplots()
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Model- Loss')
# plt.legend(['Training', 'Validation'], loc='upper right')
# plt.plot(obj.history['loss'])
# plt.plot(obj.history['val_loss'])
# plt.show()