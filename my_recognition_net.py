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
import argparse
from string import digits

def preprocess_im(image_path):
	image = image_utils.load_img(image_path)
	image = image_utils.img_to_array(image)
	image = np.expand_dims(image, axis=0)
	return image

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="images/sm_cropped/")
options = parser.parse_args()

im_dir = vars(options)['dir']

# generate dict {name => 0010...0} where 1 is on the i`th place and name is like {walrus, bear, ...}
keys = [os.path.splitext(f)[0].translate(None, digits) for f in sorted(os.listdir('images')) if f.endswith('.jpg') and not f.startswith('white1024cube')]
values = np.diag(np.ones(len(keys)))
objects = dict(zip(keys, values))

image_height = 128
image_width = 128

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
model.add(Dense(len(keys)))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(optimizer='adam', loss='mse')
model.compile(optimizer=sgd, loss='mse')

# teaching
filenames = [f for f in sorted(os.listdir(im_dir)) if f.endswith('.jpg')]
print filenames


for name in filenames:
	im = preprocess_im(im_dir + name)
	name_key = os.path.splitext(name)[0].translate(None, digits)
	y = np.asarray(objects[name_key], dtype='float')
	y = y.reshape((1,) + y.shape)
	print 'training ' + str(name) + ' ' + str(y)
	model.train_on_batch(im, y)


im = preprocess_im('images/small/bear.jpg')
predict = model.predict_on_batch(im)
print predict
print "end"
