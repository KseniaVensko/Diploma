from keras.models import load_model
import argparse
import socket_utils
from socket_utils import send_tcp_command, recv_tcp_command
from socket_utils import send_mes, recv_mes
import logger
import numpy as np
from keras.utils import np_utils
from keras.preprocessing import image as image_utils
import time
from PIL import Image
import theano
import os
theano.config.openmp = True

my_name = "pretrained_recognition"
log_file = os.path.dirname(os.path.abspath(__file__)) + '/loggers/recognition_logger.txt'
coef = 0.5
objects_count = 3
objects = {}
img_width, img_height = 128, 128

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="pretrained_vgg16_2.h5")
parser.add_argument("--port", type=int, default=7777)
parser.add_argument("--addr", type=str, default='127.0.0.1')
options = parser.parse_args()

port = vars(options)['port']
address = vars(options)['addr']
global s
global model
# keys is array of objects i.e [bear,crocodile,...]
global keys

def initialize():
	model_file = vars(options)['model']
	
	logger.write_to_log(log_file,my_name, "load sockets and model")
	global s
	s = socket_utils.initialize_client_socket(port)
	#s = socket_utils.initialize_client_socket_tcp(address, port)
	send_mes(s, "recognize_net", (address, port))
	#send_tcp_command("recognize_net", s)
	
	global model
	# will also take care of compiling the model using the saved training configuration
	model = load_model(model_file)
	
	logger.write_to_log(log_file,my_name, "sockets and model are loaded")
	global keys
	keys = []

	with open('objects.txt') as f:
			for line in f:
				keys.append(line.rstrip('\n'))
	
	logger.write_to_log(log_file,my_name, "object keys " + str(keys))
	logger.write_to_log(log_file,my_name, "initialization complete")
	print "initialization complete"

def decode_predict_proba(predict):
	# predict is like [[ 0.2, 0.99, ...]]
	predict = predict[0]
	print "predict is " + str(predict)
	summ = sum(predict)
	percentage_predict = [x / summ for x in predict]
	print "percentage predict is " + str(percentage_predict)
	# get indices of objects, that have the probability higher, than coef
	# and get maximum values
	ob = [ (n,i) for n,i in enumerate(percentage_predict) if i>coef ]
	ob.sort(key=lambda x: x[1])
	ob = ob[-objects_count:]
	seen_objects = []
	for i,n in ob:
		seen_objects.append(keys[i])
	
	return seen_objects

def preprocess_im(image_path):
	# terrible crutch for resizing images (TODO: you should to remove this or not)
	im = Image.open(image_path)
	width, height = im.size
	if width != img_width or height != img_height:
	# TODO: this is a bad practise
		print "resizing from " + str(width) + ":" + str(height) + " to " + str(img_width) + ":" + str(img_height)
		im = im.resize((img_width,img_height))
		im.save(image_path)
	im.close()
	
	image = image_utils.load_img(image_path)
	image = image_utils.img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = image.astype('float32')
	image /= 255
	
	return image

initialize()

teach_command = 'objectteaching'
teach_success = 'recognitionsuccess'
recognize_command = 'recognize'
recognize_sucess = 'seenobjects'

def teaching(path, objects):
	im = preprocess_im(path)
	y = np.zeros(len(keys))
	dict = {}
	for i,n in enumerate(keys):
		dict[n] = i
	for i in objects:
		#TODO: it is about 2 or 3 objects on image
		if not i:
			continue
		y[dict[i]] = 1
	y = y.reshape((1,-1))
	print 'training ' + str(objects)
	loss = model.train_on_batch(im,y)
	print model.metrics_names
	print loss
	
	logger.write_to_log(log_file,my_name, "train " + str(y))

while True:
	mes, addr = recv_mes(s)
	#mes = recv_tcp_command(s)
	
	logger.write_to_log(log_file,my_name, "received mes " + mes)
	
	if mes.startswith(teach_command):
		print "received teaching command"
		mes = mes.split(',')
		path = mes[1]
		teaching(path, mes[2:])
		print "sending teaching success"
		send_mes(s, teach_success, addr)
		#send_tcp_command(teach_success, s)

	elif mes.startswith(recognize_command):
		print "received recognize command"
#		send_tcp_command('waiting', s)
		send_mes(s,'waiting', addr)
		
		#path = socket_utils.receive_tcp_image(s)
		path = socket_utils.receive_image(s)
		print path
		im = preprocess_im(path)
		predict = model.predict_proba(im)
		
		logger.write_to_log(log_file,my_name, "predict " + str(predict))
		seen_objects = decode_predict_proba(predict)
		logger.write_to_log(log_file,my_name, "seen objects " + str(seen_objects))
		print "sending recognize success with objects " + str(seen_objects)
		data = recognize_sucess + ':' + ','.join(seen_objects)
		send_mes(s,data,addr)
		#send_tcp_command(data, s)
		
	elif mes.startswith('save_recognize_model'):
		mes = mes.split(',')
		path = mes[1]
		print "saving model to " + path
		model.save(path)
		send_mes(s,recognize_sucess, addr)
		#send_tcp_command(recognize_success, s)
		break

