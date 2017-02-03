import numpy as np
import os
from string import digits
import images_utils
from socket_utils import *
from PIL import Image
from polygon_actions import *
import math
import theano

image_side_size = 128
images_folder = 'images/cube_128_images/'
result_dir = 'images/two_objects/'
keys = [images_folder + f for f in sorted(os.listdir(images_folder)) if f.endswith('.jpg')]
objects_dimensions = images_utils.find_objects_hw(keys)
print keys
objects_dict = dict(zip(keys, objects_dimensions))

def get_random_images_names():
	i1 = randint(len(keys)) - 1
	im1 = keys[i1]
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
	
	print("got random images names " + im1 + ", " + im2 + ", " + im3)
	selected = [im1,im2,im3]
	coefs = [i1,i2,i3]
	return selected, coefs

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
	
	print("got random coords " + str(output))
	
	return output

def send_command_new(data, addr):
	s.sendto(data, addr)
	data, addr = s.recvfrom(1024)
	return data

s = initialize_server_socket(7777)
data, recognize_addr = s.recvfrom(1024)
print recognize_addr
#recognize_addr = ('127.0.0.1', 7777)	
for i in range(1000):
	selected, coefs = get_random_images_names()
	print selected
	x1,y1,w1,h1 = objects_dict[selected[0]]
	x2,y2,w2,h2 = objects_dict[selected[1]]
	#x3,y3,w3,h3 = objects_dict[selected[2]]
	x3 = y3 = w3 = h3 = 0
	#~ #w2=h2=w3=h3 = 0
	coords = generate_correct_random_output_coords(w1,h1,w2,h2,w3,h3)

	result_name = images_utils.draw_image(selected,image_side_size,objects_dict,result_dir, coords)
	print result_name
	objects = os.path.splitext(os.path.basename(result_name))[0].split("_")
	
	mes = send_command_new('objectteaching' + ',' + result_name + ',' + ','.join(objects), recognize_addr)

mes = send_command_new('save_recognize_model' + ',' + 'saved_model.h5', recognize_addr)
