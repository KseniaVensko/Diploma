import cv2
from scipy import ndimage
import os
from string import digits
import numpy as np
import logger
import math

# finds an object on a white background and returns coordinates of bounding rectangle
def edge_detect(file_name, tresh_min, tresh_max):
	image = cv2.imread(file_name)
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	gray = cv2.GaussianBlur(gray, (7,7),0)

	edged = cv2.Canny(gray, 50, 100)
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None,  iterations=1)

	contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
	cnt = contours[0]
	x,y,w,h = cv2.boundingRect(cnt)
	#~ cv2.rectangle(image, (x+1, y+1), (x+w-1, y+h-1), (0,255,0), 2)
	#~ cv2.imwrite(file_name, image)
	#~ cv2.imshow('lala', image)
	#~ cv2.waitKey()
 #   cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)
	
	return x,y,w,h
	
def find_objects_hw(directory):
	dimensions = []
	for filename in sorted(os.listdir(directory)):
		# (x,y) is the top-left coordinate of the rectangle
		x,y,w,h = edge_detect(directory + filename, 128, 255)
		dimensions.append([x,y,w,h])
	dimensions = np.array(dimensions, dtype='float')
	return dimensions

def putLogo(logo, canvas, x, y):
	h, w = logo.shape[:2]
	h, w = ceil_coords([h,w])
	
	logger.write_to_log("images_utils", "logo rect canvas " + str(x) + ":" + str(w+x) + ":"+ str(y) + ":"+ str(h+y))
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
	#~ cv2.rectangle(canvas, (x, y), (x+w, y+h), (255,0,0), 2)
	#~ cv2.imshow('putting_on_canvas', canvas)
	#~ cv2.waitKey(0)
	
def create_blank(width, height, rgb_color=(255, 255, 255)):
	logger.write_to_log("images_utils", "creating blank canvas " + str(height))
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

def cut_and_rotate_roi(im_name, angle, s):
	im = cv2.imread(im_name)
	# scale
	print str(int(math.ceil(128*s)))
	im = cv2.resize(im, (int(math.ceil(128*s)), int(math.ceil(128*s))))
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
	h1=w1=h2=w2=h3=w3=0
	x1, y1, a1, s1, x2, y2, a2, s2, x3, y3, a3, s3 = coords
	
	x1, y1, x2, y2, x3, y3 = ceil_coords([x1, y1, x2, y2, x3, y3])
	print 'points for drawing ' + str([x1, y1, x2, y2, x3, y3])

	h1,w1,roi1 = cut_and_rotate_roi(selected[0], a1, s1)
	
	if len(selected) > 1:
		h2,w2,roi2 = cut_and_rotate_roi(selected[1], a2, s2)
		
	if len(selected) > 2:
		h3,w3,roi3 = cut_and_rotate_roi(selected[2], a3, s3)

	res_size = max(h1+y1,w1+x1,h2+y2,w2+x2,h3+y3,w3+x3)	
	canvas = create_blank(res_size, res_size)
		
	putLogo(roi1, canvas, x1, y1)
	
	name1_clear = os.path.splitext(os.path.basename(selected[0]))[0].translate(None, digits)
	name2_clear = name3_clear = ''
	if len(selected) > 1:
		putLogo(roi2, canvas, x2, y2)
		name2_clear = os.path.splitext(os.path.basename(selected[1]))[0].translate(None, digits)
	if len(selected) > 2:
		putLogo(roi3, canvas, x3, y3)
		name3_clear = os.path.splitext(os.path.basename(selected[2]))[0].translate(None, digits)
	
	result_name = images_folder + name1_clear + '_' + name2_clear + '_' + name3_clear + '.jpg'
	
	canvas = cv2.resize(canvas, (result_size, result_size))
	
	if not os.path.exists(images_folder):
		os.makedirs(images_folder)
		
	cv2.imwrite(result_name, canvas)
	
	return result_name

