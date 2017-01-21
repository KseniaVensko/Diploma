from keras.models import load_model
import argparse
import socket_utils
import logger
import numpy as np
from keras.utils import np_utils
from keras.preprocessing import image as image_utils
import time
from PIL import Image
import theano
theano.config.openmp = True

my_name = "pretrained_recognition"
log_file = 'loggers/recognition_logger.txt'
coef = 0.3
objects_count = 3
objects = {}
img_width, img_height = 128, 128

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="pretrained_vgg16.h5")
parser.add_argument("--port", type=int, default=7777)
options = parser.parse_args()
global port
#global listening_sock, sending_sock
global s
global model
# keys is array of objects i.e [bear,crocodile,...]
global keys

def initialize():
	model_file = vars(options)['model']
	global port
	port = vars(options)['port']
	
	logger.write_to_log(log_file,my_name, "load sockets and model")
	#~ global listening_sock, sending_sock
	#~ listening_sock, sending_sock = socket_utils.initialize_sockets(port)
	global s
	s = socket_utils.initialize_client_socket(port)
	s.sendto("recognize", ('<broadcast>', port))
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
	summ = sum(predict)
	percentage_predict = [x / summ for x in predict]
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
		y[dict[i]] = 1
	y = y.reshape((1,-1))
	print 'training'
	model.train_on_batch(im,y)
	
	logger.write_to_log(log_file,my_name, "train " + str(y))

while True:
	#mes = listening_sock.recv(1024)
	mes, addr = s.recvfrom(1024)
	
	logger.write_to_log(log_file,my_name, "received mes " + mes)
	
	if mes.startswith(teach_command):
		mes = mes.split(',')
		path = mes[1]
		teaching(path, mes[2:])
		#sending_sock.sendto(teach_success, ('<broadcast>', port))
		s.sendto(teach_success, addr)

	elif mes.startswith(recognize_command):
		mes = mes.split(',')
		path = mes[1]
		im = preprocess_im(path)
		predict = model.predict_proba(im)
		
		logger.write_to_log(log_file,my_name, "predict " + str(predict))
		seen_objects = decode_predict_proba(predict)
		
		logger.write_to_log(log_file,my_name, "seen objects " + str(seen_objects))
		data = recognize_sucess + ':' + str(seen_objects)
		#sending_sock.sendto(data, ('<broadcast>', port))
		s.sendto(data, addr)

