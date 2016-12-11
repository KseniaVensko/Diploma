import numpy as np
import sys

# argument is the number of animals
n = int(sys.argv[1])
raitings = np.random.random_integers(1024, size=n*(n-1)*(n-2))
np.savetxt('raitings.txt', raitings, fmt='%u', delimiter=' ')
