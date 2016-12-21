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
	h,w,channels = logo.shape
	print h, w
	
	logger.write_to_log("images_utils", "logo rect canvas " + str(x) + ":" + str(w+x) + ":"+ str(y) + ":"+ str(h+y) + ":")
	#roi = canvas[x:w+x, y:h+y]
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
	logger.write_to_log("images_utils", "creating blank canvas " + str(height))
	image = np.zeros((height, width, 3), np.uint8)
	# Since OpenCV uses BGR, convert the color first
	color = tuple(reversed(rgb_color))
	# Fill image with color
	image[:] = color
	
	return image

# TODO: what a fuck with ceil?!
def draw_image(name1, name2, name3, result_size, objects_dict, images_folder, x1, y1, a1, x2, y2, a2, x3, y3, a3):
	h1=w1=h2=w2=h3=w3=0
	x1=int(math.ceil(x1))
	y1=int(math.ceil(y1))
	x2=int(math.ceil(x2))
	y2=int(math.ceil(y2))
	x3=int(math.ceil(x3))
	y3=int(math.ceil(y3))
	img1 = cv2.imread(name1)
	x_roi_1,y_roi_1,w_roi_1,h_roi_1 = objects_dict[name1].astype('int')
	# cut roi from image and then rotate roi and put on canvas
	# img[y: y + h, x: x + w]
	img1_roi = img1[y_roi_1:y_roi_1+h_roi_1, x_roi_1:x_roi_1+w_roi_1]
	img1_roi_rotated = ndimage.rotate(img1_roi, a1, cval=255)
	h1, w1 = img1_roi_rotated.shape[:2]
	h1=int(math.ceil(h1))
	w1=int(math.ceil(w1))
	if name2 != None:
		img2 = cv2.imread(name2)
		x_roi_2,y_roi_2,w_roi_2,h_roi_2 = objects_dict[name2].astype('int')
		img2_roi = img2[y_roi_2:y_roi_2+h_roi_2, x_roi_2:x_roi_2+w_roi_2]
		img2_roi_rotated = ndimage.rotate(img2_roi, a2, cval=255)
		h2, w2 = img2_roi_rotated.shape[:2]
		h2=int(math.ceil(h2))
		w2=int(math.ceil(w2))
	if name3 != None:
		img3 = cv2.imread(name3)
		x_roi_3,y_roi_3,w_roi_3,h_roi_3 = objects_dict[name3].astype('int')
		img3_roi = img3[y_roi_3:y_roi_3+h_roi_3, x_roi_3:x_roi_3+w_roi_3]
		img3_roi_rotated = ndimage.rotate(img3_roi, a3, cval=255)
		h3, w3 = img3_roi_rotated.shape[:2]
		h3=int(math.ceil(h3))
		w3=int(math.ceil(w3))

	res_size = max(h1+y1,w1+x1,h2+y2,w2+x2,h3+y3,w3+x3)	
	
	canvas = create_blank(res_size, res_size)
	print res_size
	print h1,w1,h2,w2,h3,w3
	logger.write_to_log("images_utils", "roi sizes of images " + str(h1) + " " + str(w1) + " "+ str(h2) + " " + str(w2) + " "+ str(h3) + " "+ str(w3))
	logger.write_to_log("images_utils", "roi coords of images " + str(y1) + " " + str(x1) + " "+ str(y2) + " " + str(x2) + " "+ str(y3) + " "+ str(x3))
	putLogo(img1_roi_rotated, canvas, x1, y1)
	if name2 != None:
		putLogo(img2_roi_rotated, canvas, x2, y2)
	if name3 != None:
		putLogo(img3_roi_rotated, canvas, x3, y3)
	
	name1_clear = os.path.splitext(os.path.basename(name1))[0].translate(None, digits)
	name2_clear = os.path.splitext(os.path.basename(name2))[0].translate(None, digits)
	name3_clear = os.path.splitext(os.path.basename(name3))[0].translate(None, digits)
	result_name = images_folder + name1_clear + '_' + name2_clear + '_' + name3_clear + '.jpg'
	canvas = cv2.resize(canvas, (result_size, result_size))
	cv2.imwrite(result_name, canvas)
	
	return result_name

