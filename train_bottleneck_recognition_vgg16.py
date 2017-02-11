import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", type=str, default=current_dir + "/images/train_dir/", help='The last slash is required')
parser.add_argument("--validation_dir", type=str, default=current_dir + "/images/validation_dir/", help='The last slash is required')
parser.add_argument("--weights_path", type=str, default=current_dir + '/vgg16_weights.h5')
parser.add_argument("--result_model_path", type=str, default=current_dir + '/pretrained_vgg16_3.h5')
parser.add_argument("--rotate", type=bool, default=False)
parser.add_argument("--imsize", type=int, default=224)
options = parser.parse_args()

train_data_dir = vars(options)['train_dir']
validation_data_dir = vars(options)['validation_dir']
weights_path = vars(options)['weights_path']
result_model_path = vars(options)['result_model_path']
imsize = vars(options)['imsize']
rotation_range = 180 if vars(options)['rotate'] else 0

top_model_weights_path = current_dir + '/bottleneck_fc_model.h5'
img_width, img_height = imsize, imsize	# in vgg16 they are 224,224

nb_epoch = 50
#class_dictionary_train=None
#class_dictionary_val=None

def find_nb_of_samples(data_dir):
	keys = [d for d in sorted(os.listdir(data_dir))]
	images_counts = [len(os.listdir(data_dir + k)) for k in keys]
	
	return sum(images_counts)

def load_vgg16_CNN_layers(img_width, img_height):
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
	
	return model

def save_bottlebeck_features():
	datagen = ImageDataGenerator(rescale=1./255, rotation_range=rotation_range)
	
	model = load_vgg16_CNN_layers(img_width, img_height)
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
	for key,value in class_dictionary_train.items():
		print(key)
		print(value)
	
	nb_train_samples = find_nb_of_samples(train_data_dir)
	bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
	np.save(open('bottleneck_features_train3.npy', 'w'), bottleneck_features_train)
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
		
	nb_validation_samples = find_nb_of_samples(validation_data_dir)
	bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
	np.save(open('bottleneck_features_validation3.npy', 'w'), bottleneck_features_validation)
	print('validation features saved')
	
	return model

def find_labels(data_dir):
	# find all classes in order like flow_from_directory uses
	keys = [d for d in sorted(os.listdir(data_dir))]
	#keys = np.unique(np.asarray(keys)).tolist()
	
	# write to file for pretrained_recognition_cnn to use
	with open('objects.txt', 'w+') as f:
		for i in keys:
			f.write(i + "\n")
	
	nb_classes = len(keys)
	values = range(nb_classes)
	train_classes = np_utils.to_categorical(values, nb_classes)
	# counts of images for each class
	images_counts = [len(os.listdir(data_dir + k)) for k in keys]
	print("images counts for " + data_dir + " are " + str(images_counts))
	train_labels = []
	for i in range(len(keys)):
		train_labels += images_counts[i] * [train_classes[i]]
	train_labels = np.asarray(train_labels)
	
	return train_labels

def initialize_fc_model(input_shape, output_shape):
	model = Sequential()
	model.add(Flatten(input_shape=input_shape))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(output_shape, activation='sigmoid'))
	
	return model

def train_top_model():
	train_data = np.load(open('bottleneck_features_train3.npy'))
#    train_labels = np.array([0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

	validation_data = np.load(open('bottleneck_features_validation3.npy'))
	nb_classes = len([d for d in sorted(os.listdir(train_data_dir))])
	model = initialize_fc_model(train_data.shape[1:], nb_classes)

	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	print 'model compiled'
	train_labels = find_labels(train_data_dir)
	validation_labels = find_labels(validation_data_dir)
	
	hist = model.fit(train_data, train_labels,
			  nb_epoch=nb_epoch, batch_size=32,
			  validation_data=(validation_data, validation_labels))
	print(hist.history)
	model.save_weights(top_model_weights_path)
	
	return train_data.shape[1:], nb_classes

save_bottlebeck_features()

input_shape, output_shape = train_top_model()
model = load_vgg16_CNN_layers(img_width, img_height)
top_model = initialize_fc_model(input_shape, output_shape)
top_model.load_weights(top_model_weights_path)

model.add(top_model)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.save(result_model_path)
