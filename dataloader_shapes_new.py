import torchvision
from PIL import Image
from torch.utils import data as data_utils
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

class Shapes_dataset(data_utils.Dataset):
    def __init__(self, test=False, dir='./data', size=(64, 64), all_files=False):
        self.datafile = dir + '/3dshapes.h5'

        if not os.path.exists(dir+'/all_x.npy'):
            self.f = h5py.File(self.datafile, 'r')
            print("HDF File opened ..")

            print("getting data ...")
            self.all_x = self.f['images']
            print("getting labels ...")
            self.all_y = self.f['labels']

            # self.all_x = np.array(self.all_x)
            # self.all_y = np.array(self.all_y)
            np.save(file=dir+'/all_x.npy', arr=self.all_x)
            np.save(file=dir+'/all_y.npy', arr=self.all_y)
            print("npy files created and saved ...")
            del self.f, self.all_x, self.all_y

        self.tensor_transform = torchvision.transforms.ToTensor()
        self.all_x = np.load(dir+'/all_x.npy')
        self.all_y = np.load(dir+'/all_y.npy')
        print("self.all_x.shape: ", self.all_x.shape)
        print("self.all_y.shape: ", self.all_y.shape)
        print("read from npy files ...")

        self.normalize_transform = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #  _FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
        #                      'orientation']
        # _NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
        #                           'scale': 8, 'shape': 4, 'orientation': 15}

    def __len__(self):
        return self.all_x.shape[0]

    def __getitem__(self, index):
        x = self.all_x[index]
        x = Image.fromarray(x)
        x = self.tensor_transform(x)
        y = self.all_y[index]
        # y = torch.Tensor(y)
        return x, y

    def get_factorVAE_data(self):
        return self.all_x, self.all_y

# dataset_test = Shapes_dataset(dir='./data', test=True, size=(64,64))
# test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=4, shuffle=False)
