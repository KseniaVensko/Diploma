import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import logger
from keras.preprocessing import image as image_utils
from keras.models import load_model

from string import digits

def preprocess_im(image_path):
	image = image_utils.load_img(image_path)
	image = image_utils.img_to_array(image)
#	image = np.expand_dims(image, axis=0)
	image = image.astype('float32')
	image /= 255
	return image

dir = 'images/evaluate_without_walrus/'
x = []
for f in sorted(os.listdir(dir)):
	im = preprocess_im(dir + f)
	print im
	x.append(im)
x = np.array(x)
#print x
keys = [os.path.splitext(os.path.basename(d))[0].translate(None, digits) for d in sorted(os.listdir(dir))]
keys = np.unique(np.asarray(keys)).tolist()
nb_classes = len(keys)
values = range(nb_classes)
train_classes = np_utils.to_categorical(values, nb_classes)
train_labels = []
for i in range(len(keys)):
	train_labels += 24 * [train_classes[i]]
train_labels = np.asarray(train_labels)

model = load_model('pretrained_vgg16.h5')
score = model.evaluate(x,train_labels)
print model.metrics_names
print score
