#from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import keras.optimizers
from keras.utils import np_utils
from keras import backend as K
import os
from string import digits
from keras.preprocessing import image as image_utils
from PIL import Image

def preprocess_im(image_path):
	image = image_utils.load_img(image_path)
	image = image_utils.img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = image.astype('float32')
	image /= 255
	return image

im_dir = 'images/without_walrus/'

# generate dict {name => 0010...0} where 1 is on the i`th place and name is like {walrus, bear, ...}
keys = [os.path.splitext(f)[0].translate(None, digits) for f in sorted(os.listdir(im_dir)) if f.endswith('.jpg')]
keys = np.unique(np.asarray(keys)).tolist()

with open('objects.txt', 'w+') as f:
	for i in keys:
		f.write(i + "\n")
		
print "written"
nb_classes = len(keys)
# TODO: 0 - should be "I don`t know" and 1:nb_classes+1 - classes of images
values = range(nb_classes)
y_train = np_utils.to_categorical(values, nb_classes)
objects = dict(zip(keys, y_train))

# input image dimensions
# TODO(1):IMPORTAINT image size from 128 to 224
# like zf net https://arxiv.org/pdf/1311.2901v3.pdf
# here is text description https://dspace.cvut.cz/bitstream/handle/10467/64667/F3-BP-2016-Jasek-Otakar-jasek.pdf
# here is about Layers 3,4,5 http://cs231n.stanford.edu/slides/winter1516_lecture7.pdf
img_rows, img_cols = 224, 224
input_shape = (3, img_rows, img_cols)
model = Sequential()
# Layer 1
model.add(Convolution2D(96, 7, 7, border_mode='valid', subsample=(2, 2), input_shape=input_shape)) # subsample is the stride here
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
# Layer 2
model.add(Convolution2D(256, 5, 5, border_mode='valid', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
# Layer 3
model.add(Convolution2D(512, 3, 3, border_mode='valid', subsample=(1, 1)))
model.add(Activation('relu'))
# Layer 4
model.add(Convolution2D(1024, 3, 3, border_mode='valid', subsample=(1, 1)))
model.add(Activation('relu'))
# Layer 5
model.add(Convolution2D(512, 3, 3, border_mode='valid', subsample=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
# Layer 6
model.add(Flatten())
model.add(Dense(4096))
model.add(Dropout(0.5))
#Layer 7
model.add(Dense(4096))
model.add(Dropout(0.5))
# Layer 8
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9)
#model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print 'Compilation of model complete'

filenames = [f for f in sorted(os.listdir(im_dir)) if f.endswith('.jpg')]

for name in filenames:
	# terrible crutch for resizing images (TODO(1))
	im = Image.open(im_dir + name)
	im = im.resize((224,224))
	im.save(im_dir + name)
	
	im = preprocess_im(im_dir + name)
	name_key = os.path.splitext(name)[0].translate(None, digits)
	y = np.asarray(objects[name_key], dtype='float')
	y = y.reshape((1,) + y.shape)
	print 'training ' + str(name) + ' ' + str(y)
	model.train_on_batch(im, y)

#~ im = preprocess_im('images/small/sheep.jpg')
#~ predict = model.predict_on_batch(im)
#~ print predict
#~ im = preprocess_im('images/small/bear.jpg')
#~ predict = model.predict_on_batch(im)
#~ print predict
model.save('recognition_model2.h5')
print "end"
#~ model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          #~ verbose=1, validation_data=(X_test, Y_test))
#~ score = model.evaluate(X_test, Y_test, verbose=0)
#~ print('Test score:', score[0])
#~ print('Test accuracy:', score[1])
