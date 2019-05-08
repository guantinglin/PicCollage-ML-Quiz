import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class RegressionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, transform=None):
        """
        :param data_folder: folder where data files are stored        
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        assert split in {'TRAIN','VAL','TEST'}
        
        self.split = split
        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + '.hdf5'), 'r')
        self.imgs = self.h['images']
        self.corrs = self.h['corrs']
        
        self.dataset_size = self.h.attrs['dataset_size']

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform


    def __getitem__(self, i):
    
        # Remember, the input pixel value of image would be assigned with 0 or 1
        img = torch.FloatTensor(self.imgs[i] / 255.)
        if self.transform is not None:
            img = self.transform(img)        
        corr = torch.FloatTensor(self.corrs[i])
        
        
        return corr,img
        
        
    def __len__(self):
        return self.dataset_size
