import os
import numpy as np
import h5py
import torch
from scipy.misc import imread, imresize
from tqdm import tqdm
import random
import csv
from collections import OrderedDict

def create_input_files(csv_path, image_folder, output_folder,
                       train_num,val_num,
                       resize = 162):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param csv_path: path of csv file with corelations of each image
    :param image_folder: folder with downloaded images
    """
    
    # for replication
    random.seed(0)
    
    # To keep the order of image names in dict (for replication)
    labels = OrderedDict()

    with open(csv_path, newline='') as csvfile:
    
        # read csv file into dictionary
        rows = csv.DictReader(csvfile)
            
        for row in rows:
            
            name = row['id']
            labels[name] = float(row['corr'])            
    
    imgs = list(labels.keys())    
    random.shuffle(imgs)     
    train_imgs = imgs[:train_num]
    val_imgs = imgs[train_num:train_num+val_num]
    test_imgs = imgs[train_num+val_num:]

    for img_paths, split in [(train_imgs,'TRAIN'),
                             (val_imgs,'VAL'), 
                             (test_imgs, 'TEST')]:
                             
        with h5py.File(os.path.join(output_folder, split + '_IMAGES_'+'.hdf5'), 'a') as h:                     
            
            images = h.create_dataset('images', (len(img_paths), 3, resize, resize), dtype='uint8')
            corrs = h.create_dataset('corrs',(len(img_paths),1),dtype='float64')
            h.attrs['dataset_size'] = len(img_paths)
            print("\nReading %s images and storing to file...\n" % split)    
            
            for i, name in enumerate(tqdm(img_paths)):
                
                # read the image with binary mode
                img = imread(image_folder+'/'+name+'.png','L')                                
                
                img = imresize(img,(resize, resize), 'nearest')                
                
                img = img[np.newaxis,:,:]
                img = np.concatenate([img, img, img], axis=0)
                images[i] = img   
                
                corrs[i] = labels[name]
            
     
     
if __name__ == '__main__':

    # the database would be split into the ratio of 
    # train:val:test = 70:20:10
    create_input_files('./train_responses.csv', './train_imgs','./data', 105000, 30000)