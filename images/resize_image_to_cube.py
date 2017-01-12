import cv2
import numpy as np
import sys
import os

im_path = sys.argv[1]
im_dir = sys.argv[2]

if not os.path.exists(im_dir):
    os.makedirs(im_dir)

h_big = 300
w_big = 300
    
im = cv2.imread(im_path)
h, w, g = im.shape
while h_big < h:
	h_big += 10
w_big = h_big
while w_big < w:
	w_big += 10
h_big = w_big

base=np.empty((h_big,w_big,3),dtype=np.uint8)
base[:] = (255,255,255)
x = (h_big - h)/2
y = (w_big - w)/2
base[x:h+x,y:w+y]=im
base = cv2.resize(base, (128,128))
im_name = os.path.basename(im_path)
print im_dir + im_name
cv2.imwrite(im_dir + im_name, base)
