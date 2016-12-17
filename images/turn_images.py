import cv2
import numpy as np
import os
import sys
from scipy import ndimage

directory = sys.argv[1]


images_paths = [f for f in sorted(os.listdir(directory)) if f.endswith('.jpg') and not f.startswith('white1024cube')]
print images_paths

for path in images_paths:
	im = cv2.imread(directory + path)
	angle = np.random.random_integers(359)
	rot = ndimage.rotate(im, angle, cval=255)
	cv2.imwrite(directory + path, rot);
