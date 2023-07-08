
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class InfantDataset(Dataset):
    def __init__(self, images_path, classes):

        self.images_path = images_path
        self.classes = classes
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(str(self.images_path[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image/255.0
        # image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image).unsqueeze(0)

        return image, self.classes[index]

    def __len__(self):
        return self.n_samples
