import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import math
import logger

log_file = 'loggers/polygon_logger.txt'
my_name = 'polygon_actions'
randint = np.random.random_integers

def check_coords_above_zero(points):
	for p in points:
		if p[0] < 0 or p[1] < 0:
			return False
	return True

def generate_polygon_from_points(h,w,coords):
	x,y,a,s = coords
	if s < 0.5:
		return False, None
	if a > 360:
		return False, None
	points = find_all_points_from_left(x, y, h, w)
	points_scaled = scale_polygon(points, s)
	points_rotated = rotate_polygon(points_scaled, a)
	if not check_coords_above_zero(points_rotated):
		return False, None
	else:
		return True, points_rotated

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

def correct_second_polygon(points1_rotated, points2_rotated):
	int_ = two_polygons_intersection(points1_rotated, points2_rotated)
	if polygon_area(points1_rotated)/2 > int_.area:
		logger.write_to_log(log_file,my_name, "second_polygon 1/2area" + str(polygon_area(points1_rotated)/2 ) + ' int_area ' + str(int_.area))
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
			logger.write_to_log(log_file,my_name, "third_polygon 1/2area2" + str(polygon_area(points2_rotated)/2 ) + ' int23_area ' + str(int_23.area))
			print 'im returning true because im thinking that ' + 'area1 ' + str(polygon_area(points1_rotated)/2) + 'is bigger than ' + str(area1)
			print
			return True
	print 'fail 3'
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
