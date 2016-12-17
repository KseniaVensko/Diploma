import cv2
import numpy as np
import sys
import os

im_path = sys.argv[1]
im_dir = sys.argv[2]

im = cv2.imread(im_path)
h, w, g = im.shape
base=np.empty((1000,1000,3),dtype=np.uint8)
base[:] = (255,255,255)
x = (1024 - h)/2
y = (1024 - w)/2
base[x:h+x,y:w+y]=im
base = cv2.resize(base, (128,128))
im_name = os.path.basename(im_path)
print im_dir + "/" + im_path
cv2.imwrite(im_dir + "/" + im_name, base)
