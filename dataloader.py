### extract the data from f5py and then use the data to train the network
### need padding the direction of the light
import torch
import numpy as np
import tqdm
import h5py
from torch.utils.data import Dataset, DataLoader

class CustomTensorDataset(Dataset):
    def __init__(self, path, num_light, num_split):
        self.dataset = None
        self.path = path
        self.num_light = num_light
        self.length = 500 * 10
        self.num_split = num_split
        self.prepare()

    def prepare(self):
        self.data_list = []
        for i in tqdm.trange(self.length):
            t = np.random.permutation(self.num_light)
            l = 0
            while l < len(t):
                r = l + self.num_split
                toadd = list(t[l:r])
                if len(toadd) < self.num_split:
                    toadd += list( t[:self.num_split - len(toadd)] )
                self.data_list.append( (i,toadd) )
                l = r

    def __len__(self):
        return len(self.data_list)
        #return len(self.dataset)//self.num_light

    def open(self):
        if self.dataset is None:
            self.f = h5py.File(self.path, 'r')
            self.dataset = self.f['cropped']

    def __getitem__(self, index):
        self.open()
        index, toadd = self.data_list[index]
        return torch.tensor(self.dataset[index*self.num_light:(index+1)*self.num_light]), torch.tensor(np.array(toadd))

def get_train_data(batch_size, num_workers, **train_kwargs):
    train_data = CustomTensorDataset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = train_loader
    return data_loader

if __name__ == '__main__':
    dataset = get_train_data(
        batch_size = 4,
        num_workers = 2,
        path = '/home/hza/data/rendering/cropped.hdf5',
        num_light=1053,
        num_split=18,
    )

    import tqdm
    b = dataset.__iter__()
    import cv2
    for i in tqdm.trange(1000):
        tmp = next(b)
        A = tmp[0].detach().numpy()
        print(tmp[1])
        for i in A:
            for j in i:
                cv2.imshow('x.jpg', j)
                cv2.waitKey(0)
                break
