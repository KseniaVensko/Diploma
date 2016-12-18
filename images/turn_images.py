import cv2
import numpy as np
import os
import sys
from scipy import ndimage

directory = sys.argv[1]
output_dir = sys.argv[2]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

images_paths = [f for f in sorted(os.listdir(directory)) if f.endswith('.jpg') and not f.startswith('white1024cube')]
print images_paths

for path in images_paths:
	for i in range(5):
		im = cv2.imread(directory + path)
		angle = np.random.random_integers(359)
		rot = ndimage.rotate(im, angle, cval=255)
		name = os.path.splitext(path)[0] + str(i)
		cv2.imwrite(output_dir + name + '.jpg', rot);
