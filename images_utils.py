import cv2
from scipy import ndimage
import os
from string import digits
import numpy as np

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

# finds an object on a white background and returns coordinates of bounding rectangle
def find_object_hw(file_path):
	# (x,y) is the top-left coordinate of the rectangle
	x,y,w,h = edge_detect(file_path, 128, 255)
	return x,y,w,h

def putLogo(logo, canvas, x, y):
    rows,cols,channels = logo.shape
    roi = canvas[x:rows+x, y:cols+y]
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
    canvas[x:rows+x, y:cols+y] = dst

def create_blank(width, height, rgb_color=(255, 255, 255)):
    image = np.zeros((height, width, 3), np.uint8)
    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

def draw_image(name1, name2, name3, result_size, images_folder, x1, y1, a1, x2, y2, a2, x3, y3, a3):
	# images are 128x128, so we need canvas with size 128*sqrt(2)x128*sqrt(2). It will be about 182x182
	# but if we want to move images to other locations, we need, for example, 2xdimensions
	canvas = create_blank(result_size, result_size)
	img1 = cv2.imread(name1)
	putLogo(ndimage.rotate(img1, a1, cval=255), canvas, x1, y1)
	if name2 != None:
		img2 = cv2.imread(name2)
		putLogo(ndimage.rotate(img2, a2, cval=255), canvas, x2, y2)
	if name3 != None:
		img3 = cv2.imread(name3)
		putLogo(ndimage.rotate(img3, a3, cval=255), canvas, x3, y3)
	name1_clear = os.path.splitext(os.path.basename(name1))[0].translate(None, digits)
	name2_clear = os.path.splitext(os.path.basename(name2))[0].translate(None, digits)
	name3_clear = os.path.splitext(os.path.basename(name3))[0].translate(None, digits)
	images_folder = 'images/small_cropped/'
	result_name = images_folder + name1_clear + '_' + name2_clear + '_' + name3_clear + '.jpg'
	canvas = cv2.resize(canvas, (128,128))
	cv2.imwrite(result_name, canvas)
	return result_name

