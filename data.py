
import os
import numpy as np
import cv2
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader

def load_names(path, file_path):
    f = open(file_path, "r")
    data = f.read().split("\n")[:-1]
    images = [os.path.join(path,"images", name) + ".jpg" for name in data]
    masks = [os.path.join(path,"masks", name) + ".jpg" for name in data]
    return images, masks

def load_data(path):
    train_names_path = f"{path}/train.txt"
    valid_names_path = f"{path}/val.txt"

    train_x, train_y = load_names(path, train_names_path)
    valid_x, valid_y = load_names(path, valid_names_path)

    return (train_x, train_y), (valid_x, valid_y)

class KvasirDataset(Dataset):
    """ Dataset for the Kvasir-SEG dataset. """
    def __init__(self, images_path, masks_path, size):
        """
        Arguments:
            images_path: A list of path of the images.
            masks_path: A list of path of the masks.
        """

        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Reading image and mask. """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        gray = cv2.imread(self.images_path[index], cv2.IMREAD_GRAYSCALE)

        """ Resizing. """
        image = cv2.resize(image, self.size)
        mask = cv2.resize(mask, self.size)
        gray = cv2.resize(gray, self.size)

        """ Proper channel formatting. """
        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)
        gray = np.expand_dims(gray, axis=0)

        """ Normalization. """
        image = image/255.0
        mask = mask/255.0
        gray = gray/255.0

        """ Changing datatype to float32. """
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)
        gray = gray.astype(np.float32)

        """ Changing numpy to tensor. """
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        gray = torch.from_numpy(gray)

        return image, mask, gray

    def __len__(self):
        return self.n_samples
