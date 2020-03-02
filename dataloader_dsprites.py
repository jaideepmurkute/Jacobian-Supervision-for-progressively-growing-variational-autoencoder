import numpy as np
import torch
from torch.utils.data import DataLoader

class Dsprites_dataset(object):
    def __init__(self, root):
        loc = root + '/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        #loaded_npz = np.load(loc)['imgs'][::3]
        print("loc: ", loc)
        self.loaded_npz = np.load(loc, encoding='latin1')['imgs']

    def __len__(self):
        # return self.imgs.size(0)
        return self.loaded_npz.shape[0]

    def __getitem__(self, index):
        # print("item: ", torch.from_numpy(self.loaded_npz[index]))
        # exit()
        return torch.from_numpy(self.loaded_npz[index]).float()


def getShapesLoader(root, batchsize, use_cuda = True, dataset_type=None):
    dataset = Dsprites_dataset(root=root)
    kwargs = {'num_workers': 0, 'pin_memory': use_cuda}
    if dataset_type == 'whole_mig':
        return DataLoader(dataset=dataset,
               batch_size=batchsize, shuffle=False, **kwargs)
    else:
        return DataLoader(dataset=dataset,
               batch_size=batchsize, shuffle=True, **kwargs)
    


# loader = getShapesLoader(root='./data', batchsize=32)
# for i, data in enumerate(loader):
#     print("i: ", i)
#     print("data.size(): ", data.size())
#     # if i == 50:
#     #     break