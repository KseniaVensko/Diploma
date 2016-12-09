import numpy as np
from shapely.geometry import Polygon
from shapely import affinity

def save_txt(inputs, outputs, file_name):
	np.savetxt(file_name + '_in', inputs, fmt='%u', delimiter=' ')
	np.savetxt(file_name + '_out', outputs, fmt='%u', delimiter=' ')

def load_txt(file_name):
	inputs = np.loadtxt(file_name + '_in', dtype=np.integer, delimiter=' ')
	outputs = np.loadtxt(file_name + '_out', dtype=np.integer, delimiter=' ')
	return inputs, outputs

def save_float_txt(output, file_name):
	np.savetxt(file_name + '_out', output, fmt='%.2f', delimiter=' ')

def save_to_txt(inputs, outputs, file_name):
	ar = np.append(inputs, outputs, 1)
	np.savetxt(file_name, ar, fmt='%u', delimiter=' ')
	
def load_from_txt(file_name):
	a = np.loadtxt(file_name, dtype=np.integer, delimiter=' ')
	return a

def correct_second_polygon(coords, h1, w1, h2, w2):
	if coords[3] == 0:
		return True
	points1 = find_all_points_from_left(coords[0], coords[1], h1, w1)
	points2 = find_all_points_from_left(coords[3], coords[4], h2, w2)
	
	points1_rotated = rotate_polygon(points1, coords[2])
	points2_rotated = rotate_polygon(points2, coords[5])
	
	int_area = two_polygons_intersection_area(points1_rotated, points2_rotated)
	
	if polygon_area(points1)/2 > int_area:
		return True
	return False

def correct_third_polygon(coords, h1, w1, h2, w2, h3, w3):
	if coords[6] == 0:
		return True
	points1 = find_all_points_from_left(coords[0], coords[1], h1, w1)
	points2 = find_all_points_from_left(coords[3], coords[4], h2, w2)
	points3 = find_all_points_from_left(coords[6], coords[7], h3, w3)
	
	points1_rotated = rotate_polygon(points1, coords[2])
	points2_rotated = rotate_polygon(points2, coords[5])
	points3_rotated = rotate_polygon(points2, coords[8])

	int_area13 = two_polygons_intersection_area(points1_rotated, points3_rotated)
	int_area12 = two_polygons_intersection_area(points1_rotated, points2_rotated)
	int_area23 = two_polygons_intersection_area(points2_rotated, points3_rotated)
	area = int_area12 + int_area13 - int_area23
	
	if polygon_area(points1)/2 > area:
		return True
	return False

# Shoelace formula for calculating the area of the polygon
# from here: stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
def poly_area(x,y):
	return 0.5*np.abs(np.dot(x,np.roll(y,1)) - np.dot(y,np.roll(x,1)))

def polygon_area(points):
	p = Polygon(points)
	return p.area

def two_polygons_intersection_area(points1, points2):
	p = Polygon(points1)
	p2 = Polygon(points2)
	x = p.intersection(p2)
	return x.area
	
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
