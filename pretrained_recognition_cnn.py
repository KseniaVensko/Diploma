from keras.models import load_model
import argparse
import socket_utils
import logger
import numpy as np
from keras.utils import np_utils
from keras.preprocessing import image as image_utils
import time

my_name = "pretrained_recognition"
coef = 0.8
objects_count = 3
objects = {}

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="recognition_model.h5")
parser.add_argument("--port", type=int, default=7777)
options = parser.parse_args()
global port
global listening_sock, sending_sock
global model, keys

def initialize():
	model_file = vars(options)['model']
	global port
	port = vars(options)['port']

	logger.write_to_log(my_name, "load sockets and model")
	global listening_sock, sending_sock
	listening_sock, sending_sock = socket_utils.initialize_sockets(port)
	global model
	model = load_model(model_file)
	logger.write_to_log(my_name, "sockets and model are loaded")
	global keys
	keys = []

	with open('objects.txt') as f:
			for line in f:
				keys.append(line.rstrip('\n'))
	#~ nb_classes = len(keys)
	#~ values = range(nb_classes)
	#~ val = np_utils.to_categorical(values, nb_classes)
	#~ objects = dict(zip(keys, val))

	logger.write_to_log(my_name, "object keys" + str(keys))
	logger.write_to_log(my_name, "initialization complete")
	print "initialization complete"

def decode_predictions(predict):
	# predict is like [[ 0.2, 0.99, ...]]
	predict = np.asarray(predict[0])
	# get indices of objects, that have the probability higher, than coef
	# and get maximum values
	ob = [ (n,i) for n,i in enumerate(predict) if i>coef ]
	ob.sort(key=lambda x: x[1])
	ob = ob[-objects_count:]
	seen_objects = []
	for i,n in ob:
		seen_objects.append(keys[i])
	
	return seen_objects

def preprocess_im(image_path):
	image = image_utils.load_img(image_path)
	image = image_utils.img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = image.astype('float32')
	image /= 255
	
	return image

initialize()
while True:
	mes = listening_sock.recv(1024)
	logger.write_to_log(my_name, "received mes " + mes)
	if mes.startswith('objectteaching'):
		mes = mes.split(',')
		x = np.asarray(map(int, mes[1:4]))
		y = np.asarray(map(int, mes[4:]))
		teaching(model, x, y)
		data = 'recognitionsuccess'
		sending_sock.sendto(data, ('<broadcast>', port))

	elif mes.startswith('recognize'):
		mes = mes.split(',')
		path = mes[1]
		im = preprocess_im(path)
		predict = model.predict_on_batch(im)
		logger.write_to_log(my_name, "predict " + str(predict))
		seen_objects = decode_predictions(predict)
		logger.write_to_log(my_name, "seen objects " + str(seen_objects))
		data = 'seenobjects:' + str(seen_objects)
		sending_sock.sendto(data, ('<broadcast>', port))

