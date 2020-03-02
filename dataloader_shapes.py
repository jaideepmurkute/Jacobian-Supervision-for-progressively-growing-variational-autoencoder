import torchvision
from PIL import Image
from torch.utils import data as data_utils
import glob
import os
from numpy import genfromtxt
from torchvision.utils import save_image


class Shapes_dataset(data_utils.Dataset):
    def __init__(self, test=False, dir='./data', size=(64, 64), all_files=False):
        if all_files:
            print("Collecting all the files (train + test) ...")
            self.files = os.listdir(dir + '/train/')
            print("Number of train files: ", len(self.files))
            for i in range(len(self.files)):
                self.files[i] = dir + '/train/' + self.files[i]
            self.files_test = os.listdir(dir + '/test/')
            print("Number of test files: ", len(self.files_test))
            for i in range(len(self.files_test)):
                self.files.append(dir + '/test/' + self.files_test[i])
        else:
            if not test:
                print("Collecting train files ...")
                self.files = os.listdir(dir + '/train/')
                for i in range(len(self.files)):
                    self.files[i] = dir + '/train/' + self.files[i]
            if test:
                print("Collecting test files ...")
                self.files = os.listdir(dir + '/test/')
                for i in range(len(self.files)):
                    self.files[i] = dir + '/test/' + self.files[i]

        print("Creating dataset with {} files ...".format(len(self.files)))
        self.size = (size, size) if type(int) else size
        self.tensor_transform = torchvision.transforms.ToTensor()
        self.resize_transform = torchvision.transforms.Resize(self.size[0], interpolation=2)
        self.labels = genfromtxt('labels.txt', delimiter=',')
        # two braces are desired means and std, after normalization, for RGB channels.
        self.normalize_transform = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # print(self.files[index])
        x = Image.open(self.files[index])
        x = self.resize_transform(x)
        x = self.tensor_transform(x)
        # print(x)
        n = self.files[index] + ''
        n = n.replace('.png', '')
        n = int(n.split('/')[-1])

        y = self.labels[n]

        return x, y

#
# #TESTER
# dataset_test = Shapes_dataset(dir='./Shapes/', test=True, size=(64,64))
# dataset_train = Shapes_dataset(dir='./Shapes/', test=False, size=(64,64))
#
# import torch
# train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=4, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=4, shuffle=False)
#
# for batch, (x, y) in enumerate(train_loader):
#     save_image(x[0].view(1, 3, 64, 64), 'test.png')
#     print("ok")
#     break
