import torch
import argparse
import glob
import cv2
import tqdm
import os
import h5py
from torch import nn
import numpy as np

class Dataset:
    def __init__(self, path='/home/hza/data/rendering/orgTrainingImages/Shape_Multi_500_full/train/imgs'):
        self.path = path
        dataFolder = self.path
        num = 0
        for i in sorted(glob.glob(os.path.join(self.path, 'Shape__*'))):
            num += 1
        files = [os.path.join(self.path, 'Shape__{}'.format(i), '0', 'inters') for i in range(500)]
        img_paths = []
        for i in range(num):
            tmp = []
            for j in sorted(os.listdir(files[i])):
                tmp.append(os.path.join(files[i], j))
            img_paths.append(tmp)
        self.imgs = img_paths
        with open('crop_ids.txt', 'r') as f:
            self.crop_ids = [list(map(int, i.strip().split())) for i in f.readlines()]
        print(len(self.crop_ids))

    def createF5py(self, num=10, size=128, path='/home/hza/data/rendering/cropped.hdf5'):
        num_scene = len(self.imgs)
        num_light = len(self.imgs[0])
        total = num_scene * num_light * num
        if os.path.exists(path):
            print('file already exists')
            return

        cc = 0
        with h5py.File(path, 'w') as f:
            dset = f.create_dataset('cropped', (total, size, size, 3), dtype='uint8')

            for i in tqdm.trange(len(self.imgs)):
                tmps = self.imgs[i]
                coors = []
                for k in range(num):
                    coor = self.crop_ids[cc]
                    coors.append(coor)
                    cc += 1

                for iters, j in enumerate(tmps):
                    img = cv2.imread(j)
                    for k in range(num):
                        a, b = coors[k]
                        cropped = img[a:a+128, b:b+128]
                        if size != 128:
                            cropped = cv2.resize(cropped, (size, size))
                        dset[(i * num + k)*num_light + iters] = cropped

def play():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', default=64)
    args = parser.parse_args()

    dataset = Dataset()
    dataset.createF5py(size=args.size, path='/home/hza/data/rendering/cropped_{}.hdf5'.format(args.size))

if __name__ == '__main__':
    play()