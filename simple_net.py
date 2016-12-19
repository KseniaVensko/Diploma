from keras.models import Sequential
from keras.layers import Dense
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error
import math
import keras.layers.advanced_activations as aa
import socket
import cv2
from scipy import ndimage
import argparse
import os
from string import digits
import images_utils
import socket_utils

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="images/training_im/")
parser.add_argument("--result", type=str, default="images/result/result.jpg")
parser.add_argument("--port", type=int, default=7777)
options = parser.parse_args()

images_folder = vars(options)['dir']
result_name = vars(options)['result']
port = vars(options)['port']

# images are 128x128, so we need canvas with size 128*sqrt(2)x128*sqrt(2). It will be about 182x182
# but if we want to move images to other locations, we need, for example, 2xdimensions
blank_side_size = 182*2
#scaler = MinMaxScaler(feature_range=(0,1))

def initialize_model():
	epoch_size = 64*4
	inputs = 3
	model = Sequential()
	# input: h1/w1 h2/w2 h3/w3
	model.add(Dense(12, init='normal', input_dim=inputs, activation='relu'))
	model.add(Dense(12, init='normal'))
	model.add(Dense(9, init='normal', activation='relu'))
	# output: x1 y1 a1 x2 y2 a2 x3 y3 a3

	model.compile(optimizer='adam', loss='mse')
	return model

def teaching(model, x, y):
	x = x.reshape((1,-1))
	y = y.reshape((1,-1))
	print 'teaching ' + str(x) + ' ' + str(y)
	#model.train_on_batch(x, y)
	return model
	
def get_images_names():
	# TODO: get 3 different objects
	images = [(images_folder + f) for f in sorted(os.listdir(images_folder)) if f.endswith('.jpg')]
	return images[0], images[1], images[2]
	
def get_coordinates(name1, name2, name3):
	#~ im1 = cv2.imread(name1)
	#~ h1, w1, g = im1.shape
	#~ im2 = cv2.imread(name2)
	#~ h2, w2, g = im2.shape
	#~ im3 = cv2.imread(name3)
	#~ h3, w3, g = im3.shape
	# TODO: preprocess
	x = np.asarray([1, 1.3, 0.95])
	print x
	#predict = model.predict_on_batch(x)
	# TODO: check that coordinates are correct
	y = np.asarray([1,2,45,3,4,20,1,1,45])
	print y
	return y

model = initialize_model()
listening_sock, sending_sock = socket_utils.initialize_sockets(port)
print 'initialization complete'

while True:
	mes = listening_sock.recv(1024)
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
		# TODO: check that image.size is (128, 128)
		x1,y1,a1,x2,y2,a2,x3,y3,a3 = get_coordinates(n1, n2, n3)
		# TODO: check that coordinates are correct
		result_name = images_utils.draw_image(n1,n2,n3,blank_side_size,images_folder,x1,y1,a1,x2,y2,a2,x3,y3,a3)
		data = 'imagegenerated,' + result_name
		sending_sock.sendto(data, ('<broadcast>', port))

#x = np.random.random_integers(1024, size=6)
#y = np.random.random_integers(360, size=9)
#teaching(model, x, y)
#x = x.reshape((1,-1))
#predict = model.predict_on_batch(x)
#print predi


#train_on_batch(self, x, y, class_weight=None, sample_weight=None)

#score = model.evaluate(x_test, y_test, batch_size=1)
