import os
import numpy as np

image_files = [f for f in sorted(os.listdir('.')) if f.endswith('.jpg') and not f.startswith('white1024cube')]
np.savetxt('image_descriptors', image_files, fmt='%s', delimiter=' ')
