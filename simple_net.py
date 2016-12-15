from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from polygon_actions import load_txt, save_txt, save_float_txt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import keras.layers.advanced_activations as aa
import socket
import cv2
from scipy import ndimage

white_cube = 'images/white1024cube.jpg'
images_folder = 'images/'

def initialize_model():
	epoch_size = 64*4
	model = Sequential()
	# input: h1 w1 h2 w2 h3 w3
	model.add(Dense(12, init='normal', input_dim=6, activation='relu'))
	#model.add(aa.LeakyReLU(alpha=0.3))
	model.add(Dense(12, init='normal'))
	model.add(Dense(9, init='normal', activation='relu'))
	# output: x1 y1 a1 x2 y2 a2 x3 y3 a3

	model.compile(optimizer='adam', loss='mse')
	return model

def initialize_sockets(port):
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
	s.bind(('0.0.0.0', port))

	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST,1)
	return s, sock

def teaching(model, x, y):
	x = x.reshape((1,-1))
	y = y.reshape((1,-1))
	print x
	print y
	model.train_on_batch(x, y)
	return model
	
def putLogo(logo, img1, x, y):
    rows,cols,channels = logo.shape
    roi = img1[x:rows+x, y:cols+y]
    # Now create a mask of logo and create its inverse mask
    logogray = cv2.cvtColor(logo,cv2.COLOR_BGR2GRAY)
    # add a threshold
    ret, mask = cv2.threshold(logogray, 220, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    # Take only region of logo from logo image.
    logo_fg = cv2.bitwise_and(logo,logo,mask = mask)
    dst = cv2.add(img1_bg,logo_fg)
    img1[x:rows+x, y:cols+y] = dst

def draw_image(name1, name2, name3, result_name, x1, y1, a1, x2, y2, a2, x3, y3, a3):
	canvas = cv2.imread(white_cube)
	img1 = cv2.imread(name1)
	putLogo(ndimage.rotate(img1, 10, cval=255), canvas, x1, y1)
	if name2 != None:
		img2 = cv2.imread(name2)
		putLogo(ndimage.rotate(img2, a2, cval=255), canvas, x2, y2)
	if name3 != None:
		img3 = cv2.imread(name3)
		putLogo(ndimage.rotate(img3, 0, cval=255), canvas, x3, y3)
	cv2.imwrite(images_folder + result_name, canvas);
		
def get_images_names():
	return 'images/bear.jpg', 'images/sheep.jpg', 'images/walrus.jpg'
	
def get_coordinates(name1, name2, name3):
	im1 = cv2.imread(name1)
	h1, w1, g = im1.shape
	im2 = cv2.imread(name2)
	h2, w2, g = im2.shape
	im3 = cv2.imread(name3)
	h3, w3, g = im3.shape
	x = np.asarray([889, 742, 472, 598, 878, 720])
	# here should be predict = model.predict_on_batch(x)
	y = np.random.random_integers(360, size=9)
	print y
	return y

model = initialize_model()
listening_sock, sending_sock = initialize_sockets(7777)

while True:
#for i in range(10):
	mes = listening_sock.recv(1024)
	if mes.startswith('teaching'):
		mes = mes.split(',')
		x = np.asarray(map(int, mes[1:7]))		
		y = np.asarray(map(int, mes[7:]))
		teaching(model, x, y)
	elif mes.startswith('generate'):
		n1,n2,n3 = get_images_names()
		x1,y1,a1,x2,y2,a2,x3,y3,a3 = get_coordinates(n1, n2, n3)
		draw_image(n1,n2,n3, 'result.jpg', x1,y1,a1,x2,y2,a2,x3,y3,a3)

#x = np.random.random_integers(1024, size=6)
#y = np.random.random_integers(360, size=9)
#teaching(model, x, y)
#x = x.reshape((1,-1))
#predict = model.predict_on_batch(x)
#print predi


#train_on_batch(self, x, y, class_weight=None, sample_weight=None)

#score = model.evaluate(x_test, y_test, batch_size=1)
