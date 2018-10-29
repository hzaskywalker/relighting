import os
import numpy as np
with open(os.path.join(os.path.dirname(__file__), 'Coefs.txt'), 'r') as f:
    dirs = f.readlines()[:1053]

dirs = [list( map(float, i.strip().split(' ')) ) for i in dirs]
dirs = np.array(dirs)
