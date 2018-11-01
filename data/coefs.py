import os
import numpy as np

with open(os.path.join(os.path.dirname(__file__), 'Coefs.txt'), 'r') as f:
    coefs = f.readlines()[:1053]

coefs = [list( map(float, i.strip().split(' ')) ) for i in coefs]
coefs = np.array(coefs)


with open(os.path.join(os.path.dirname(__file__), 'Dirs.txt'), 'r') as f:
    dirs = f.readlines()[:1053]

dirs = [list( map(float, i.strip().split(' ')) ) for i in dirs]
dirs = np.array(dirs)

length = (dirs[:,:2] ** 2).sum(axis=1) ** 0.5
angles = np.arctan2(length, dirs[:,2])

dirs = dirs[:,:2]