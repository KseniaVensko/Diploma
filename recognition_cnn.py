#from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import os
from string import digits
from keras.preprocessing import image as image_utils

def preprocess_im(image_path):
	image = image_utils.load_img(image_path)
	image = image_utils.img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = image.astype('float32')
	image /= 255
	return image

batch_size = 128

nb_epoch = 12

# input image dimensions
img_rows, img_cols = 128, 128
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
input_shape = (3, img_rows, img_cols)
im_dir = 'images/training_im/'

# generate dict {name => 0010...0} where 1 is on the i`th place and name is like {walrus, bear, ...}
keys = [os.path.splitext(f)[0].translate(None, digits) for f in sorted(os.listdir(im_dir)) if f.endswith('.jpg')]
keys = np.unique(np.asarray(keys)).tolist()
nb_classes = len(keys)
# TODO: 0 - should be "I don`t know" and 1:nb_classes+1 - classes of images
values = range(nb_classes)
y_train = np_utils.to_categorical(values, nb_classes)
objects = dict(zip(keys, y_train))

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

filenames = [f for f in sorted(os.listdir(im_dir)) if f.endswith('.jpg')]
for name in filenames:
	im = preprocess_im(im_dir + name)
	name_key = os.path.splitext(name)[0].translate(None, digits)
	y = np.asarray(objects[name_key], dtype='float')
	y = y.reshape((1,) + y.shape)
	print 'training ' + str(name) + ' ' + str(y)
	model.train_on_batch(im, y)

im = preprocess_im('images/sm_cropped/kangoro50.jpg')
predict = model.predict_on_batch(im)
print predict
print "end"
#~ model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          #~ verbose=1, validation_data=(X_test, Y_test))
#~ score = model.evaluate(X_test, Y_test, verbose=0)
#~ print('Test score:', score[0])
#~ print('Test accuracy:', score[1])
