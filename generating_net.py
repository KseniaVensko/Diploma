from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error
from keras import initializations
import argparse
import os
from string import digits
import images_utils
import socket_utils
import logger
from PIL import Image
from polygon_actions import *

my_name="generating_net"
current_y_sequence = []
current_x_sequence = []

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="images/big_without_walrus/")
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
object_coefs = None
threshold = 0.8
locate_model = None
selecting_model = None
keys = []

def my_init(shape, name=None):
    return initializations.uniform(shape, scale=0.5, name=name)

def resize_images_to_standard_size(paths):
	for path in paths:
		im = Image.open(path)
		width, height = im.size
		if width != image_side_size or height != image_side_size:
			im = im.resize((image_side_size, image_side_size))
			im.save(path)
		im.close()

def initialize_models():
	logger.write_to_log(my_name, "initialize objects_dict")
	global keys
	keys = [images_folder + f for f in sorted(os.listdir(images_folder)) if f.endswith('.jpg')]
	
	# TODO: maybe this is a bad practise
	resize_images_to_standard_size(keys)
	
	objects_dimensions = images_utils.find_objects_hw(images_folder)
	objects_dict = dict(zip(keys, objects_dimensions))
	logger.write_to_log(my_name, "initialization of objects_dict complete")
	
	objects_count = len(keys)
	global object_coefs
	object_coefs = np.random.random_integers(100, size=objects_count)
	logger.write_to_log(my_name, "starting object_coefs " + str(object_coefs))
	
	selecting_model = Sequential()
	# for start let take 1+objects_count inputs - last for random number and one for each object coefficient
	selecting_model.add(Dense(8, input_dim=objects_count+1, activation='relu'))
	selecting_model.add(Dense(8, activation='relu'))
	selecting_model.add(Dense(objects_count, activation='softmax'))
	# output is like [ 0.2, 0.99, ... ] and we take only those which is more than treshold and then choose 3
	selecting_model.compile(optimizer='adam', loss='mse')
	
	# divide h/w
	b = np.divide(objects_dimensions[:,3],objects_dimensions[:,2])
	b = b.reshape(-1,1)
	scaler.fit(b)
	epoch_size = 64*4
	inputs = 3
	locate_model = Sequential()
	# input: h1/w1 h2/w2 h3/w3
	locate_model.add(Dense(12, init=my_init, input_dim=inputs, activation='relu'))
	locate_model.add(Dense(12, init=my_init, activation='relu'))
	locate_model.add(Dense(9, init=my_init, activation='relu'))
	# output: x1 y1 a1 x2 y2 a2 x3 y3 a3
	locate_model.compile(optimizer='adam', loss='mse')
	
	logger.write_to_log(my_name, "initialization of models complete")
	return locate_model, selecting_model, objects_dict

def teaching(locate_model, x, y):
	logger.write_to_log(my_name, 'teaching ' + str(x) + ' ' + str(y))
	x = x.reshape((1,-1))
	y = y.reshape((1,-1))
	print 'teaching ' + str(x) + ' ' + str(y)
	locate_model.train_on_batch(x, y)
	return locate_model

def decode_select_predict(predict):
	# predict is like [[ 0.2, 0.99, ...]]
	predict = predict[0]
	# get indices of objects, that have the probability higher, than coef
	# and get maximum values
	ob = [ (n,i) for n,i in enumerate(predict) if i>threshold ]
	ob.sort(key=lambda x: x[1])
	objects_count = 3
	ob = ob[-objects_count:]
	selected_objects = []
	for i,n in ob:
		selected_objects.append(keys[i])
	
	return selected_objects

def get_random_images_names():
	i1 = np.random.random_integers(len(keys)) - 1
	im1 = keys[i1]
	# TODO: I removed check_image_shape because it doesnot matter, test it
	i2 = np.random.random_integers(len(keys)) - 1
	im2 = keys[i2]
	while im1.translate(None, digits) == im2.translate(None, digits):
		i2 = np.random.random_integers(len(keys)) - 1
		im2 = keys[i2]
	i3 = np.random.random_integers(len(keys)) - 1
	im3 = keys[i3]
	while im1.translate(None, digits) == im3.translate(None, digits) or im2.translate(None, digits) == im3.translate(None, digits):
		i3 = np.random.random_integers(len(keys)) - 1
		im3 = keys[i3]
	logger.write_to_log(my_name, "got random images names " + im1 + ", " + im2 + ", " + im3)
	selected = [im1,im2,im3]
	return selected
	
def get_images_names():
	r = np.random.random_integers(100)
	x = np.copy(object_coefs)
	x = np.append(x,r)
	x = x.reshape((1,) + x.shape)
	predict = selecting_model.predict(x)
	selected_objects = decode_select_predict(predict)
	logger.write_to_log(my_name, "got selected images names " + str(selected_objects))
	if len(selected_objects) < 2:
		selected_objects = get_random_images_names()
	# TODO: refactor
	return selected_objects[0], selected_objects[1], selected_objects[2]

#TODO: check correctness because its not	
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
	
	logger.write_to_log(my_name, "got random coords " + str(output))
	
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
	global current_x_sequence
	current_x_sequence = x
	# normalisation
#	x = scaler.transform(x)
	x = x.reshape(1,-1)
	predict = locate_model.predict_on_batch(x)
	predict = predict[0]
	logger.write_to_log(my_name, "predicted coords " + str(predict))
	
	if not check_if_correct(predict,h1,w1,h2,w2,h3,w3):
		predict = generate_correct_random_output_coords(w1,h1,w2,h2,w3,h3)
#	predict = scaler.inverse_transform(predict)
	global current_y_sequence
	current_y_sequence = predict
	return predict

# TODO: why not working the line below
#if __name__ == 'main':
locate_model, selecting_model, objects_dict = initialize_models()
listening_sock, sending_sock = socket_utils.initialize_sockets(port)
print 'initialization complete'

while True:
	mes = listening_sock.recv(1024)
	logger.write_to_log(my_name, "received mes " + mes)
	# simplenetteaching,h1/w1,h2/w2,h3/w3,x1,y1,a1,x2,y2,a2,x3,y3,a3
	if mes.startswith('generatingnetteaching'):
		#~ mes = mes.split(',')
		#~ x = np.asarray(map(int, mes[1:4]))
		#~ y = np.asarray(map(int, mes[4:]))
		x = current_x_sequence
		y = current_y_sequence
		teaching(locate_model, x, y)
		data = 'generatingnetsuccess'
		sending_sock.sendto(data, ('<broadcast>', port))

	elif mes.startswith('generate'):
		n1,n2,n3 = get_images_names()
		x1,y1,a1,x2,y2,a2,x3,y3,a3 = get_coordinates(n1, n2, n3)
		result_name = images_utils.draw_image(n1,n2,n3,image_side_size,objects_dict,'images/result2/', x1,y1,a1,x2,y2,a2,x3,y3,a3)
		logger.write_to_log(my_name, "name of result image " + result_name)
		data = 'imagegenerated,' + result_name
		sending_sock.sendto(data, ('<broadcast>', port))
