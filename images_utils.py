import cv2
from scipy import ndimage
import os
from string import digits
import numpy as np
import logger
import math

image_side_size = 512
log_file = 'loggers/images_utils_logger.txt'

# finds an object on a white background and returns coordinates of bounding rectangle
def edge_detect(file_name, tresh_min, tresh_max):
	image = cv2.imread(file_name)
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	#gray = cv2.GaussianBlur(gray, (5,5),0)
	gray = cv2.medianBlur(gray,5)    

	edged = cv2.Canny(gray, 10, 200)
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None,  iterations=1)

	contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	try: hierarchy = hierarchy[0]
	except: hierarchy = []

	height, width = edged.shape[:2]
	min_x, min_y = width, height
	max_x = max_y = 0
	for contour, hier in zip(contours, hierarchy):
		(x,y,w,h) = cv2.boundingRect(contour)
		min_x, max_x = min(x, min_x), max(x+w, max_x)
		min_y, max_y = min(y, min_y), max(y+h, max_y)
	#~ if max_x - min_x > 0 and max_y - min_y > 0:
		#~ cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
	
	w = max_x - min_x
	h = max_y - min_y
	x = min_x
	y = min_y
	#cv2.imwrite(file_name, image)
	
	return x,y,w,h
	
def find_objects_hw(images_array):
	dimensions = []
	for filename in images_array:
		# (x,y) is the top-left coordinate of the rectangle
		x,y,w,h = edge_detect(filename, 128, 255)
		dimensions.append([x,y,w,h])
	dimensions = np.array(dimensions, dtype='float')
	return dimensions

def putLogo(logo, canvas, x, y):
	h, w = logo.shape[:2]
	h, w = ceil_coords([h,w])
	
	logger.write_to_log(log_file,"images_utils", "logo rect canvas " + str(x) + ":" + str(w+x) + ":"+ str(y) + ":"+ str(h+y))
	roi = canvas[y:h+y, x:w+x]
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
	canvas[y:h+y, x:w+x] = dst
	
def create_blank(width, height, rgb_color=(255, 255, 255)):
	logger.write_to_log(log_file,"images_utils", "creating blank canvas " + str(height))
	image = np.zeros((height, width, 3), np.uint8)
	# Since OpenCV uses BGR, convert the color first
	color = tuple(reversed(rgb_color))
	# Fill image with color
	image[:] = color
	
	return image

def ceil_coords(coords):
	for c in coords:
		c = int(math.ceil(c))
	coords = map(int, coords)
	
	return coords

# TODO: maybe should take the size of image, not the standard size
def cut_and_rotate_roi(im_name, angle, s):
	im = cv2.imread(im_name)
	# scale
	im = cv2.resize(im, (int(math.ceil(image_side_size*s)), int(math.ceil(image_side_size*s))))
	temp_name = os.path.splitext(im_name)[0] + '_temp.jpg'
	cv2.imwrite(temp_name, im)
	
	# coords of roi
	x,y,w,h = edge_detect(temp_name, 128,255)
	os.remove(temp_name)
	im_roi = im[y:y+h,x:x+w]
	# relative to the center
	im_roi_rotated = ndimage.rotate(im_roi, angle, cval=255)
	h, w = im_roi_rotated.shape[:2]
	h, w = ceil_coords([h,w])
	
	return h,w,im_roi_rotated

def draw_image(selected, result_size, objects_dict, images_folder, coords):
	h1=w1=h2=w2=h3=w3=x2=y2=a2=s2=x3=y3=a3=s3=0
	if len(selected) > 2:
		x1, y1, a1, s1, x2, y2, a2, s2, x3, y3, a3, s3 = coords
	elif len(selected) > 1:
		x1, y1, a1, s1, x2, y2, a2, s2 = coords
	else:
		x1, y1, a1, s1 = coords
	# TODO: this is coorinates of turned and scaled polygon, but I use them as they are only scaled
	x1, y1, x2, y2, x3, y3 = ceil_coords([x1, y1, x2, y2, x3, y3])
	print 'points for drawing ' + str([x1, y1, x2, y2, x3, y3])

	h1,w1,roi1 = cut_and_rotate_roi(selected[0], a1, s1)
	
	if len(selected) > 1:
		h2,w2,roi2 = cut_and_rotate_roi(selected[1], a2, s2)
		
	if len(selected) > 2:
		# TODO: this is a stub, think about selected coords: zeros or length? and see also line 127
		if x3 != y3 != a3 != s3 != 0:
			print "third object"
			h3,w3,roi3 = cut_and_rotate_roi(selected[2], a3, s3)

	res_size = max(h1+y1,w1+x1,h2+y2,w2+x2,h3+y3,w3+x3)
	# TODO: coords are on the result_size^2 cube, right? - then what about res_size
	if res_size < result_size:
		canvas = create_blank(result_size, result_size)
	else:
		print "I shouldn`t be here, because coords should be in [0..result_size] interval"
		canvas = create_blank(res_size, res_size)
		
	putLogo(roi1, canvas, x1, y1)
	
	name1_clear = os.path.splitext(os.path.basename(selected[0]))[0].translate(None, digits)
	name2_clear = name3_clear = ''
	if len(selected) > 1:
		putLogo(roi2, canvas, x2, y2)
		name2_clear = os.path.splitext(os.path.basename(selected[1]))[0].translate(None, digits)
	if len(selected) > 2:
		if x3 != y3 != a3 != s3 != 0:
			putLogo(roi3, canvas, x3, y3)
			name3_clear = os.path.splitext(os.path.basename(selected[2]))[0].translate(None, digits)
	
	result_name = images_folder + '_'.join([name1_clear, name2_clear, name3_clear]) + '.jpg'
	
	canvas = cv2.resize(canvas, (result_size, result_size))
	print "result size is " + str(canvas.size)
	
	if not os.path.exists(images_folder):
		os.makedirs(images_folder)
		
	cv2.imwrite(result_name, canvas)
	
	return result_name

