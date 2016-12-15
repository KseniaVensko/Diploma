import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
import cv2
from keras.preprocessing import image as image_utils
import os

def preprocess_im(image_path):
	image = image_utils.load_img(image_path)
	image = image_utils.img_to_array(image)
#	image = np.expand_dims(image, axis=0)
#	image = preprocess_input(image)
	return image

image_height = 128 
image_width = 128
objects_count = 7

model = Sequential()
model.add(Convolution2D(64,3,3, input_shape=(3,image_height,image_width), border_mode='valid'))
# now model.output_shape == (None, 64, image_height, image_width)
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32,3,3, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(objects_count))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(optimizer='adam', loss='mse')
model.compile(optimizer=sgd, loss='mse')

# teaching
files = np.loadtxt('images/image_descriptors', dtype='str', delimiter=' ')
print files
for i in range(files.size):
	im = preprocess_im('images/small/' + files[i])
	im = im.reshape( (1,) + im.shape )
	y = np.zeros(7)
	y[i] = 1
	y = y.reshape((1,) + y.shape)
	model.train_on_batch(im, y)

im = preprocess_im('images/small/dog.jpg')
im = im.reshape( (1,) + im.shape )
predict = model.predict_on_batch(im)
print predict
print "end"
