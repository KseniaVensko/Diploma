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
import math
import theano
theano.config.openmp = True

randint = np.random.random_integers
log_file = 'loggers/generating_net_logger.txt'
my_name="generating_net"
# for locate_net teaching
current_y_sequence = []
current_x_sequence = []
# for selecting_net teaching
current_name_input_sequence = []
current_name_sequence = []
object_coefs = None
# for decreasing or encreasing object_coefs
atom = 10
# for choosing names in prediction of selecting_net
threshold = 0.8

image_side_size = 128
# will be used for sequence preprocessing
max_dimension = 1
scaler = MinMaxScaler(feature_range=(0,1))
objects_dimensions = np.asarray([])
objects_dict = {}

locate_model = None
selecting_model = None
keys = []

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="images/big_without_walrus/")
parser.add_argument("--result_dir", type=str, default="images/result_dir/")
parser.add_argument("--port", type=int, default=7777)
options = parser.parse_args()

images_folder = vars(options)['dir']
result_dir = vars(options)['result_dir']
port = vars(options)['port']

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
	
	logger.write_to_log(log_file,my_name, "initialize objects_dict")
	global keys
	keys = [images_folder + f for f in sorted(os.listdir(images_folder)) if f.endswith('.jpg')]
	
	with open('objects.txt', 'w') as f:
		for k in keys:
			f.write(os.path.splitext(os.path.basename(k))[0].translate(None, digits) + '\n')
	# TODO: maybe this is a bad practise
	resize_images_to_standard_size(keys)
	
	objects_dimensions = images_utils.find_objects_hw(images_folder)
	objects_dict = dict(zip(keys, objects_dimensions))
	
	logger.write_to_log(log_file,my_name, "initialization of objects_dict complete")
	
	objects_count = len(keys)
	global object_coefs
	object_coefs = randint(100, size=objects_count)
	
	logger.write_to_log(log_file,my_name, "starting object_coefs " + str(object_coefs))
	
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
	locate_model.add(Dense(12, init=my_init, activation='relu'))
	# output: x1 y1 a1 s1 x2 y2 a2 s2 x3 y3 a3 s3
	locate_model.compile(optimizer='adam', loss='mse')
	
	logger.write_to_log(log_file,my_name, "initialization of models complete")
	return locate_model, selecting_model, objects_dict

def teaching(locate_model, selecting_model, x, y, input_names, names):
	
	logger.write_to_log(log_file,my_name, 'teaching locate_model ' + str(x) + ' ' + str(y))
	logger.write_to_log(log_file,my_name, 'teaching selecting_model ' + str(input_names) + ' ' + str(names))
	x = x.reshape((1,-1))
	y = y.reshape((1,-1))
	names = names.reshape((1,-1))
	input_names = input_names.reshape((1,-1))
	print 'teaching locate_model ' + str(x) + ' ' + str(y)
	print 'teaching selecting_model ' + str(input_names) + ' ' + str(names)
	locate_model.train_on_batch(x, y)
	selecting_model.train_on_batch(input_names, names)
	
	return locate_model, selecting_model

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
	selected_coef = []
	for i,n in ob:
		selected_coef.append(i)
		selected_objects.append(keys[i])
	
	return selected_objects, selected_coef

def get_random_images_names():
	i1 = randint(len(keys)) - 1
	im1 = keys[i1]
	# TODO: I removed check_image_shape because it doesnot matter, test it
	i2 = randint(len(keys)) - 1
	im2 = keys[i2]
	while im1.translate(None, digits) == im2.translate(None, digits):
		i2 = randint(len(keys)) - 1
		im2 = keys[i2]
	i3 = randint(len(keys)) - 1
	im3 = keys[i3]
	while im1.translate(None, digits) == im3.translate(None, digits) or im2.translate(None, digits) == im3.translate(None, digits):
		i3 = randint(len(keys)) - 1
		im3 = keys[i3]
	
	logger.write_to_log(log_file,my_name, "got random images names " + im1 + ", " + im2 + ", " + im3)
	selected = [im1,im2,im3]
	coefs = [i1,i2,i3]
	return selected, coefs
	
def get_images_names():
	r = randint(100)
	x = np.copy(object_coefs)
	x = np.append(x,r)
	# TODO: maybe all current sequences save after x.reshape
	global current_name_input_sequence
	current_name_input_sequence = x
	
	logger.write_to_log(log_file,my_name, "current name input sequence " + str(current_name_input_sequence))
	
	x = x.reshape((1,) + x.shape)
	predict = selecting_model.predict(x)
	selected_objects, coefs = decode_select_predict(predict)
	
	logger.write_to_log(log_file,my_name, "got selected images names " + str(selected_objects))
	if len(selected_objects) < 2:
		selected_objects, coefs = get_random_images_names()
	# TODO: refactor
	
	global current_name_sequence
	current_name_sequence = np.zeros(shape=len(keys), dtype=np.float)
	for i in coefs:
		current_name_sequence[i] = 1
	
	logger.write_to_log(log_file,my_name, "current name sequence " + str(current_name_sequence))
	
	return selected_objects

#TODO: check correctness because its not	
def generate_correct_random_output_coords(w1,h1,w2,h2,w3,h3):
	output = np.zeros(shape=12, dtype=np.float)
	x1,y1,a1,s1,points_rotated1 = generate_polygon(w1,h1,image_side_size,True)
	output[:4] = x1,y1,a1,s1
	correct = False
	if w2 != 0 and h2 != 0:
		if w3 != 0 and h3 != 0:
			while not correct:
				x1,y1,a1,s1,points_rotated1 = generate_polygon(w1,h1,image_side_size,True)
				x2,y2,a2,s2,points_rotated2 = generate_polygon(w2,h2, image_side_size)
				x3,y3,a3,s3,points_rotated3 = generate_polygon(w3,h3,image_side_size)
				correct = correct_third_polygon(points_rotated1, points_rotated2, points_rotated3)
			output[:4] = x1,y1,a1,s1
			output[4:8] = x2,y2,a2,s2
			output[8:] = x3,y3,a3,s3
		else:
			while not correct:
				x1,y1,a1,s1,points_rotated1 = generate_polygon(w1,h1,image_side_size)
				x2,y2,a2,s2,points_rotated2 = generate_polygon(w2,h2, image_side_size)
				correct = correct_second_polygon(points_rotated1, points_rotated2)
			output[:4] = x1,y1,a1,s1
			output[4:8] = x2,y2,a2,s2
	
	logger.write_to_log(log_file,my_name, "got random coords " + str(output))
	
	return output

def check_if_correct(coords,h1,w1,h2,w2,h3,w3):
	correct, points1_rotated = generate_polygon_from_points(h1,w1,coords[:4])
	points2_rotated = points3_rotated = None
	if not correct:
		return False
	if h2 != 0 and w2 != 0:
		correct, points2_rotated = generate_polygon_from_points(h2,w2,coords[4:8])
		if not correct:
			return False
	if h3 != 0 and w3 != 0:
		correct, points3_rotated = generate_polygon_from_points(h3,w3,coords[8:])
		if not correct:
			return False
	if points2_rotated != None:
		if points3_rotated != None:
			if correct_third_polygon(points1_rotated, points2_rotated, points3_rotated):
				return True
			else:
				return False
		else:
			if correct_second_polygon(points1_rotated, points2_rotated):
				return True
			else:
				return False

def get_coordinates(selected):
	x1,y1,w1,h1 = objects_dict[selected[0]]
	w2=h2=w3=h3 = 0
	
	x = [h1/w1]
	if len(selected) > 1:
		x2,y2,w2,h2 = objects_dict[selected[1]]
		x.append(h2/w2)
	else:
		x.append(0)
	if len(selected) > 2:
		x3,y3,w3,h3 = objects_dict[selected[2]]
		x.append(h3/w3)
	else:
		x.append(0)
	
	logger.write_to_log(log_file,my_name, "current x sequence " + str(x))
	x = np.asarray(x)
	global current_x_sequence
	current_x_sequence = x
	# normalisation
#	x = scaler.transform(x)
	x = x.reshape(1,-1)
	predict = locate_model.predict_on_batch(x)
	predict = predict[0]
	
	logger.write_to_log(log_file,my_name, "----------------")
	logger.write_to_log(log_file,my_name, "predicted coords " + str(predict))
	logger.write_to_log(log_file,my_name, "----------------")
	
	if not check_if_correct(predict,h1,w1,h2,w2,h3,w3):
		predict = generate_correct_random_output_coords(w1,h1,w2,h2,w3,h3)
#	predict = scaler.inverse_transform(predict)
	# TODO: ceil coords for better teaching
	global current_y_sequence
	current_y_sequence = predict
	
	logger.write_to_log(log_file,my_name, "current y sequence " + str(current_y_sequence))
	return predict

# TODO: why not working the line below
#if __name__ == 'main':
locate_model, selecting_model, objects_dict = initialize_models()
#listening_sock, sending_sock = socket_utils.initialize_sockets(port)
s = socket_utils.initialize_client_socket(port)
print 'initialization complete'

teach_command = 'generatingnetteaching'
teach_success = 'generatingnetsuccess'
generate_command = 'generate'
generate_sucess = 'imagegenerated'

s.sendto("generate_net", ('<broadcast>', port))
while True:
	mes, addr = s.recvfrom(1024)
#	mes = listening_sock.recv(1024)
	
	logger.write_to_log(log_file,my_name, "received mes " + mes)
	if mes.startswith(teach_command):
		print 'received teaching command'
		success = mes.split(',')[1]
		if success == 'False':
			# TODO: decrease object_coefs
			print 'not success'
		else:
			# TODO: encrease object_coefs
			x = current_x_sequence
			y = current_y_sequence
			names = current_name_sequence
			input_names = current_name_input_sequence
			locate_model, selecting_model = teaching(locate_model, selecting_model, x, y, input_names, names)
		
		print "sending teaching success"
		s.sendto(teach_success, addr)
		#sending_sock.sendto(teach_success, ('<broadcast>', port))

	elif mes.startswith(generate_command):
		selected = get_images_names()
		# TODO: if len(selected)=2 you should take it into account when you will teach
		coords = get_coordinates(selected)
		print
		print 'selected coords ' + str(coords)
		print
		result_name = images_utils.draw_image(selected,image_side_size,objects_dict,result_dir, coords)
		
		logger.write_to_log(log_file,my_name, "name of result image " + result_name)
		print "sending generate success"
		data = generate_sucess + ',' + result_name
		s.sendto(data, addr)
		mes,addr = s.recvfrom(1024)
		if 'waiting' in mes:
			print "sending image "
			socket_utils.send_image(result_name, addr, s)
			
		#sending_sock.sendto(data, ('<broadcast>', port))
		
