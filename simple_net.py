from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error
import math
import keras.layers.advanced_activations as aa
from keras import initializations
import socket
import cv2
from scipy import ndimage
import argparse
import os
from string import digits
import images_utils
import socket_utils
from polygon_actions import *
import logger

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="images/other/")
parser.add_argument("--result", type=str, default="images/result/result.jpg")
parser.add_argument("--port", type=int, default=7777)
options = parser.parse_args()

images_folder = vars(options)['dir']
result_name = vars(options)['result']
port = vars(options)['port']

image_side_size = 128
# will be used for sequence preprocessing
max_dimension = 1
scaler = MinMaxScaler(feature_range=(0,1))
objects_dimensions = np.asarray([])
objects_dict = {}

def my_init(shape, name=None):
    return initializations.uniform(shape, scale=0.5, name=name)

def initialize_model():
	logger.write_to_log("simple_net", "initialize objects_dict")
	keys = [images_folder + f for f in sorted(os.listdir(images_folder)) if f.endswith('.jpg')]
	objects_dimensions = images_utils.find_objects_hw(images_folder)
	objects_dict = dict(zip(keys, objects_dimensions))
	logger.write_to_log("simple_net", "initialization of objects_dict complete")
	# divide h/w
	b = np.divide(objects_dimensions[:,3],objects_dimensions[:,2])
	b = b.reshape(-1,1)
	scaler.fit(b)
	epoch_size = 64*4
	inputs = 3
	model = Sequential()
	# input: h1/w1 h2/w2 h3/w3
	model.add(Dense(12, init=my_init, input_dim=inputs, activation='relu'))
	model.add(Dense(12, init=my_init))
	model.add(Dense(9, init=my_init, activation='relu'))
	# output: x1 y1 a1 x2 y2 a2 x3 y3 a3

	model.compile(optimizer='adam', loss='mse')
	logger.write_to_log("simple_net", "initialization of model complete")
	return model, objects_dict

def teaching(model, x, y):
	logger.write_to_log("simple_net", 'teaching ' + str(x) + ' ' + str(y))
	x = x.reshape((1,-1))
	y = y.reshape((1,-1))
	print 'teaching ' + str(x) + ' ' + str(y)
	#model.train_on_batch(x, y)
	return model

def check_image_shape(im_path):
	img = cv2.imread(im_path)
	h, w, c = img.shape
	logger.write_to_log("simple_net", "checking image size " + "h " + str(h) + "w " + str(w))
	if h != image_side_size or w != image_side_size or c != 3:
		return False
	return True
	
def get_images_names():
	# TODO: get 3 different objects more elegantly
	# TODO: images can be sheep3 and sheep4. Why? I fixed it, but not sure if now it is correct
	images = [f for f in sorted(os.listdir(images_folder)) if f.endswith('.jpg')]
	i1 = np.random.random_integers(len(images)) - 1
	im1 = images[i1]
	while not check_image_shape(images_folder + images[i1]):
		i1 = np.random.random_integers(len(images)) - 1
		im1 = images[i1]
	i2 = np.random.random_integers(len(images)) - 1
	im2 = images[i2]
	while im1.translate(None, digits) == im2.translate(None, digits) or not check_image_shape(images_folder + images[i2]):
		i2 = np.random.random_integers(len(images)) - 1
		im2 = images[i2]
	i3 = np.random.random_integers(len(images)) - 1
	im3 = images[i3]
	while not check_image_shape(images_folder + images[i3]) or im1.translate(None, digits) == im3.translate(None, digits) or im2.translate(None, digits) == im3.translate(None, digits):
		i3 = np.random.random_integers(len(images)) - 1
		im3 = images[i3]
	logger.write_to_log("simple_net", "got images names " + images_folder + images[i1] + ", " + images_folder + images[i2] + ", " + images_folder + images[i3])
	
	return images_folder + images[i1], images_folder + images[i2], images_folder + images[i3]
	
def generate_correct_random_output_coords(w1,h1,w2,h2,w3,h3):
	output = np.zeros(shape=9, dtype=np.float)
	output[0] = np.random.random_integers(image_side_size - w1) - 1	# x1
	output[1] = np.random.random_integers(image_side_size - h1) - 1	# y1
	output[2] = np.random.random_integers(360) - 1				# angle1
	
	output[3] = 0 if w2 == 0 else np.random.random_integers(image_side_size - w2) - 1
	output[4] = 0 if w2 == 0 else np.random.random_integers(image_side_size - h2) - 1
	output[5] = 0 if w2 == 0 else np.random.random_integers(360) - 1
	while (not correct_second_polygon(output[:6], h1, w1, h2, w2)):
		output[3] = 0 if w2 == 0 else np.random.random_integers(image_side_size - w2) - 1
		output[4] = 0 if w2 == 0 else np.random.random_integers(image_side_size - h2) - 1
		output[5] = 0 if w2 == 0 else np.random.random_integers(360) - 1

	output[6] = 0 if w3 == 0 else np.random.random_integers(image_side_size - w3) - 1
	output[7] = 0 if w3 == 0 else np.random.random_integers(image_side_size - h3) - 1
	output[8] = 0 if w3 == 0 else np.random.random_integers(360) - 1
	while (not correct_third_polygon(output, h1, w1, h2, w2, h3, w3)):
		output[6] = 0 if w3 == 0 else np.random.random_integers(image_side_size - w3) - 1
		output[7] = 0 if w3 == 0 else np.random.random_integers(image_side_size - h3) - 1
		output[8] = 0 if w3 == 0 else np.random.random_integers(360) - 1
	
	logger.write_to_log("simple_net", "got random coords " + str(output))
	
	return np.asarray(output)

def check_if_correct(coords,h1,w1,h2,w2,h3,w3):
	if (correct_second_polygon(coords[:6], h1, w1, h2, w2)
		and correct_third_polygon(coords, h1, w1, h2, w2, h3, w3)):
		return True
	return False
	
def get_coordinates(name1, name2, name3):
	x1,y1,w1,h1 = objects_dict[name1]
	x2,y2,w2,h2 = objects_dict[name2]
	x3,y3,w3,h3 = objects_dict[name3]
	x = np.asarray([h1/w1, h2/w2, h3/w3])
	# normalisation
#	x = scaler.transform(x)
	x = x.reshape(1,-1)
	#print 'x ', x
	predict = model.predict_on_batch(x)
	predict = predict[0]
	logger.write_to_log("simple_net", "predicted coords " + str(predict))
	
	if not check_if_correct(predict,h1,w1,h2,w2,h3,w3):
		predict = generate_correct_random_output_coords(w1,h1,w2,h2,w3,h3)
	#print predict
#	predict = scaler.inverse_transform(predict)
#	y = np.asarray([1,2,45,3,4,20,1,1,45])
	print predict
	return predict

# TODO: why not working the line below
#if __name__ == 'main':
model, objects_dict = initialize_model()
listening_sock, sending_sock = socket_utils.initialize_sockets(port)
print 'initialization complete'

while True:
	mes = listening_sock.recv(1024)
	logger.write_to_log("simple_net", "received mes " + mes)
	# simplenetteaching,h1/w1,h2/w2,h3/w3,x1,y1,a1,x2,y2,a2,x3,y3,a3
	if mes.startswith('simplenetteaching'):
		mes = mes.split(',')
		x = np.asarray(map(int, mes[1:4]))
		y = np.asarray(map(int, mes[4:]))
		teaching(model, x, y)
		data = 'simplenetsuccess'
		sending_sock.sendto(data, ('<broadcast>', port))

	elif mes.startswith('generate'):
		n1,n2,n3 = get_images_names()
		x1,y1,a1,x2,y2,a2,x3,y3,a3 = get_coordinates(n1, n2, n3)
		#coord = get_coordinates(n1, n2, n3)
		#x1,y1,a1,x2,y2,a2,x3,y3,a3 = coord[:9]
		result_name = images_utils.draw_image(n1,n2,n3,image_side_size,objects_dict,'images/result/', x1,y1,a1,x2,y2,a2,x3,y3,a3)
		logger.write_to_log("simple_net", "name of result image " + result_name)
		data = 'imagegenerated,' + result_name
		sending_sock.sendto(data, ('<broadcast>', port))


#train_on_batch(self, x, y, class_weight=None, sample_weight=None)

#score = model.evaluate(x_test, y_test, batch_size=1)
