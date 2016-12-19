import numpy as np
from PIL import Image
from shapely.geometry import Point
from shapely.geometry import Polygon
import argparse
import os
import random

def cut_polygon(name, region, im_pixels, im_copy, output_directory):
	for index, pixel in np.ndenumerate(im_pixels):
	  # Unpack the index.
	  row, col, channel = index
	  # We only need to look at spatial pixel data for one of the four channels.
	  if channel != 0:
		continue
	  point = Point(row, col)
	  if not region.contains(point):
		im_copy[(row, col, 0)] = 255
		im_copy[(row, col, 1)] = 255
		im_copy[(row, col, 2)] = 255
		im_copy[(row, col, 3)] = 0
		
	cut_image = Image.fromarray(im_copy)
	cut_image.save(output_directory + name + str(i) + '.jpg')

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def get_polygon_points_new(low_x, high_x, low_y, high_y):
	exclude_y = 0.2*(high_y - low_y)
	numbers_y = range(int(low_y), int(low_y + exclude_y)) + range(int(high_y - exclude_y), int(high_y))
	exclude_x = 0.2*(high_x - low_x)
	numbers_x = range(int(low_x), int(low_x + exclude_x)) + range(int(high_x - exclude_x), int(high_x))
	angles = np.random.random_integers(2) + 2
	x = y = 0
	points = np.empty(2, dtype='int')
	points = np.asarray([[random.choice(numbers_x), random.choice(numbers_y)]])
	while len(points) < angles:
		x = random.choice(numbers_x)
		y = random.choice(numbers_y)
		points = np.vstack((points, [x,y]))
		points = unique_rows(points)
		#print points
	return points

def get_sides(w,h,l,r,u,b):
	low_x = 0 + u*h
	low_y = 0 + l*w
	high_x = h - b*h
	high_y = w - r*w
	return low_x, high_x, low_y, high_y

parser = argparse.ArgumentParser(description="lala")
parser.add_argument("--name", type=str, default="bear.jpg")
parser.add_argument("--l", type=float, default=0, help="percentage to crop from left")
parser.add_argument("--r", type=float, default=0, help="lrbu")
parser.add_argument("--u", type=float, default=0, help="lrbu")
parser.add_argument("--b", type=float, default=0, help="lrbu")
parser.add_argument("--area", type=float, default=0.5)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--count", type=int, default=10)
parser.add_argument("--out_dir", type=str, default='training_images/')
options = parser.parse_args()
l = vars(options)['l']
r = vars(options)['r']
u = vars(options)['u']
b = vars(options)['b']
name = vars(options)['name']
area_coefficient = vars(options)['area']
start_index = vars(options)['start']
count = vars(options)['count']
output_directory = vars(options)['out_dir']

if not os.path.exists(output_directory):
    os.makedirs(output_directory)


im = Image.open(name).convert('RGBA')
im_pixels = np.array(im)
im_copy = np.array(im)
w,h = im.size
low_x, high_x, low_y, high_y = get_sides(w,h,l,r,u,b)

i=start_index
# y changes by -->
# x changes by |
#			   V
while True:
	if i > start_index + count:
		break
	points = get_polygon_points_new(low_x, high_x, low_y, high_y)
	region = Polygon(points)
	if region.area < area_coefficient*w*h:
		continue

	print 'points ' + str(points)
	print 'area ' + str(region.area/(w*h))
	cut_polygon(os.path.splitext(os.path.basename(name))[0], region, im_pixels, im_copy, output_directory)
	i+=1
