import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import logger

my_name = 'train_bottleneck_vgg16'
weights_path = 'vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
img_width, img_height = 128, 128	# in vgg16 they are 224,224
train_data_dir = 'images/without_walrus_alot/'
validation_data_dir = 'images/without_walrus/'
nb_train_samples = 4196
nb_validation_samples = 461
nb_epoch = 50
class_dictionary_train=None
class_dictionary_val=None

def save_bottlebeck_features():
	datagen = ImageDataGenerator(rescale=1./255)

	# build the VGG16 network
	model = Sequential()
	model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

	model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
	f = h5py.File(weights_path)
	for k in range(f.attrs['nb_layers']):
		if k >= len(model.layers):
			# we don't look at the last (fully-connected) layers in the savefile
			break
		g = f['layer_{}'.format(k)]
		weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
		model.layers[k].set_weights(weights)
	f.close()
	print('Model loaded.')
	
	# flow_from_directory uses sorted(os.listdir(directory))
	generator = datagen.flow_from_directory(
			train_data_dir,
			target_size=(img_width, img_height),
			batch_size=32,
			class_mode=None,
			shuffle=False)
	global class_dictionary_train
	class_dictionary_train = generator.class_indices
	for keys,values in class_dictionary_train.items():
		print(keys)
		print(values)
	bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
	np.save(open('bottleneck_features_train2.npy', 'w'), bottleneck_features_train)
	print('training features saved')

	generator = datagen.flow_from_directory(
			validation_data_dir,
			target_size=(img_width, img_height),
			batch_size=32,
			class_mode=None,
			shuffle=False)
	global class_dictionary_val
	class_dictionary_val = generator.class_indices
	for keys,values in class_dictionary_val.items():
		print(keys)
		print(values)
	bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
	np.save(open('bottleneck_features_validation2.npy', 'w'), bottleneck_features_validation)
	print('validation features saved')

def find_labels(data_dir):
	# find all classes in order like flow_from_directory uses
	keys = [d for d in sorted(os.listdir(data_dir))]
	keys = np.unique(np.asarray(keys)).tolist()
	nb_classes = len(keys)
	values = range(nb_classes)
	train_classes = np_utils.to_categorical(values, nb_classes)
	# counts of images for each class
	images_counts = [len(os.listdir(data_dir + k)) for k in keys]
	logger.write_to_log(my_name, "images counts for " + data_dir + " are " + str(images_counts))
	train_labels = []
	for i in range(len(keys)):
		train_labels += images_counts[i] * [train_classes[i]]
	train_labels = np.asarray(train_labels)
	return train_labels

def train_top_model():
	train_data = np.load(open('bottleneck_features_train.npy'))
#    train_labels = np.array([0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

	validation_data = np.load(open('bottleneck_features_validation.npy'))

	model = Sequential()
	model.add(Flatten(input_shape=train_data.shape[1:]))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(7, activation='sigmoid'))

	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
	print 'model compiled'
	train_labels = find_labels(train_data_dir)
	validation_labels = find_labels(validation_data_dir)
	
	model.fit(train_data, train_labels,
			  nb_epoch=nb_epoch, batch_size=32,
			  validation_data=(validation_data, validation_labels))
	model.save_weights(top_model_weights_path)    

train_top_model()