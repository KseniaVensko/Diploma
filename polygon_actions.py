import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import math
import logger

my_name = 'polygon_actions'
randint = np.random.random_integers

def check_coords_above_zero(points):
	for p in points:
		if p[0] < 0 or p[1] < 0:
			return False
	return True

def generate_polygon(w,h, image_side_size, is_first=False):
	check = False
	max_scale = 1.2
	if w > image_side_size/1.5 and h > image_side_size/1.5 and not is_first:
		max_scale = 1
	while not check:
		a = randint(360) - 1
		s = np.random.uniform(0.5,1.2,1)[0]
		x = randint(image_side_size - w) - 1
		y = randint(image_side_size - h) - 1
		
		points = find_all_points_from_left(x, y, h, w)
		points_scaled = scale_polygon(points, s)
		points_rotated = rotate_polygon(points_scaled, a)
		check = check_coords_above_zero(points_rotated)
	print list(points_rotated)
	print x,y,a,s
	return x,y,a,s,points_rotated

def check_correct_polygon(coords, h1, w1):
	x1, y1, a1, s1 = coords
	print 'in check_correct coords are' + str(coords)
	points1 = find_all_points_from_left(x1, y1, h1, w1)
	points1_scaled = scale_polygon(points1, s1)
	points1_rotated = rotate_polygon(points1_scaled, a1)
	print list(points1_rotated)
	if not check_coords_above_zero(points1_rotated):
		print 'points are below zero 1'
		return False
	return True

def correct_polygon(w1,h1, image_side_size):
	a1 = randint(90) - 1
	s1 = np.random.uniform(0.5,2,1)[0]
	scaled_w1 = w1*s1
	scaled_h1 = h1*s1
	H1 = int(math.ceil(scaled_w1 * math.sin(math.radians(a1)) + scaled_h1 * math.cos(math.radians(a1))))
	W1 = int(math.ceil(scaled_h1 * math.sin(math.radians(a1)) + scaled_w1 * math.cos(math.radians(a1))))
	while H1 > image_side_size or W1 > image_side_size:
		a1 = randint(90) - 1
		s1 = np.random.uniform(0.5,2,1)[0]
		scaled_w1 = w1*s1
		scaled_h1 = h1*s1
		H1 = int(math.ceil(scaled_w1 * math.sin(math.radians(a1)) + scaled_h1 * math.cos(math.radians(a1))))
		W1 = int(math.ceil(scaled_h1 * math.sin(math.radians(a1)) + scaled_w1 * math.cos(math.radians(a1))))
	
	print H1,W1
	print a1
	print scaled_h1, scaled_w1
	x1 = randint(0, image_side_size - W1)
	y1 = randint(math.ceil(scaled_w1 * math.sin(math.radians(a1))), image_side_size - (H1 - math.ceil(scaled_w1 * math.sin(math.radians(a1)))))
	
	print str([x1,y1,a1,s1])
	
	if not check_correct_polygon([x1,y1,a1,s1], h1, w1):
		print 'screw u'
	
	x1_new = x1
	y1_new = y1 - int(math.ceil(scaled_w1 * math.sin(math.radians(a1))))
	print 'new x and y ' + str(x1_new) + ' ' + str(y1_new)
	return x1_new, y1_new, a1, s1

def correct_second_polygon(points1_rotated, points2_rotated):
	int_ = two_polygons_intersection(points1_rotated, points2_rotated)
	if polygon_area(points1_rotated)/2 > int_.area:
		logger.write_to_log(my_name, "second_polygon 1/2area" + str(polygon_area(points1_rotated)/2 ) + ' int_area ' + str(int_.area))
		return True
	print 'fail 2'
	return False

def correct_third_polygon(points1_rotated, points2_rotated, points3_rotated):
	int_13 = two_polygons_intersection(points1_rotated, points3_rotated)
	int_12 = two_polygons_intersection(points1_rotated, points2_rotated)
	int_23 = two_polygons_intersection(points2_rotated, points3_rotated)
	
	if int_12.area == 0 or int_13.area == 0:
		int_123 = 0
	else:
		int_123_p = two_polygons_intersection(int_12.exterior.coords, points3_rotated)
		int_123 = int_123_p.area

	area1 = int_12.area + int_13.area - int_123
	
	if polygon_area(points1_rotated)/2 > area1:
		if polygon_area(points2_rotated)/2 > int_23.area:
			logger.write_to_log(my_name, "third_polygon 1/2area2" + str(polygon_area(points2_rotated)/2 ) + ' int23_area ' + str(int_23.area))
			print 'im returning true because im thinking that ' + 'area1 ' + str(polygon_area(points1_rotated)/2) + 'is bigger than ' + str(area1)
			print
			return True
	print 'fail 3'
	print
	return False

def correct_third_polygon_old(coords, h1, w1, h2, w2, h3, w3):
	print 'im in correct third polygon'
	print coords
	if h3 == 0 or w3 == 0:
		print 'h is 0'
		print
		return True
	x1, y1, a1, s1, x2, y2, a2, s2, x3, y3, a3, s3 = coords
	points1 = find_all_points_from_left(x1, y1, h1, w1)
	points2 = find_all_points_from_left(x2, y2, h2, w2)
	points3 = find_all_points_from_left(x3, y3, h3, w3)	
	print 'points'
	print str(points1)
	print str(points2)
	print str(points3)
	
	points1_scaled = scale_polygon(points1, s1)
	points2_scaled = scale_polygon(points2, s2)
	points3_scaled = scale_polygon(points3, s3)

	points1_rotated = rotate_polygon(points1_scaled, a1)
	points2_rotated = rotate_polygon(points2_scaled, a2)
	points3_rotated = rotate_polygon(points3_scaled, a3)
	
	if not check_coords_above_zero(points3_rotated):
		print 'points are below zero 3'
		return False
	
	print 'rotated points'
	print str(points1_rotated)
	print str(points2_rotated)
	print str(points3_rotated)
	int_13 = two_polygons_intersection(points1_rotated, points3_rotated)
	int_12 = two_polygons_intersection(points1_rotated, points2_rotated)
	int_23 = two_polygons_intersection(points2_rotated, points3_rotated)
		
	print 'area1 ' + str(polygon_area(points1)) + ' area2 ' + str(polygon_area(points2))
	print 'area12 ' + str(int_12.area) + ' area13 ' + str(int_13.area) + ' area23 ' + str(int_23.area)
	
	if int_12.area == 0 or int_13.area == 0:
		int_123 = 0
	else:
		print '12 coords inters ' + str(list(int_12.exterior.coords))
		print '13 coords inters ' + str(list(int_13.exterior.coords))
		print '23 coords inters ' + str(list(int_23.exterior.coords))
		int_123_p = two_polygons_intersection(int_12.exterior.coords, points3_rotated)
		int_123 = int_123_p.area

#	int_area123 = two_polygons_intersection_area
	area1 = int_12.area + int_13.area - int_123
	print 'area1/2 ' + str(polygon_area(points1)/2) + ' area_int ' + str(area1)
	#~ print 'area2/2 ' + str(polygon_area(points2)/2) + ' area23 ' + str(int_23.area)
	if polygon_area(points1)/2 > area1:
		if polygon_area(points2)/2 > int_23.area:
			return True
			print 'im returning true because im thinking that ' + 'area1 ' + str(polygon_area(points1)/2) + 'is bigger than ' + str(area1)
			print
	print 'return false'
	print
	return False

def polygon_area(points):
	p = Polygon(points)
	return p.area

def two_polygons_intersection(points1, points2):
	p = Polygon(points1)
	p2 = Polygon(points2)
	x = p.intersection(p2)
	#print x
	return x

def scale_polygon(points, fact):
	p = Polygon(points)
	scaled_p = affinity.scale(p, fact, fact)
	return list(scaled_p.exterior.coords)
	
def rotate_polygon(points, angle):
	p = Polygon(points)
	rotated_p = affinity.rotate(p, angle, points[0])
	return list(rotated_p.exterior.coords)
	
def find_all_points_from_left(x, y, h, w):
	points = []
	points.append((x,y))
	points.append((x, y+h))
	points.append((x+w, y+h))
	points.append((x+w, y))
	return points
