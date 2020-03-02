import numpy as np
import torch
from torch.utils.data import DataLoader

class Dsprites_dataset(object):
    def __init__(self, root):
        loc = root + '/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        #loaded_npz = np.load(loc)['imgs'][::3]
        self.loaded_npz = np.load('data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
                              encoding='latin1')['imgs']

    def __len__(self):
        # return self.imgs.size(0)
        return self.loaded_npz.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.loaded_npz[index])


def getShapesLoader(root, batchsize, use_cuda = True):
    dataset = Dsprites_dataset(root=root)
    kwargs = {'num_workers': 4, 'pin_memory': use_cuda}

    return DataLoader(dataset=dataset,
               batch_size=batchsize, shuffle=True, **kwargs)


loader = getShapesLoader(root='./data', batchsize=32)
for i, data in enumerate(loader):
    print("i: ", i)
    print("data.size(): ", data.size())
    # if i == 50:
    #     break
