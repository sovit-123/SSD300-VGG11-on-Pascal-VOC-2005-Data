import torch
import json
import os

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils import transform

class PascalVOCDataset(Dataset):
    """
    Custom dataset to load PascalVOC data as batches
    """
    def __init__(self, data_folder, split):
        """
        :param data_folder: folder path of the data files
        :param split: either `TRAIN` or `TEST`
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder

        # read the data files
        with open(os.path.join(data_folder, 
                               self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, 
                               self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        # read image
        image = Image.open(self.images[i])
        image = image.convert('RGB')

        # get bounding boxes, labels, diffculties for the corresponding image
        # all of them are objects
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)

        # apply transforms
        image, boxes, labels = transform(image, boxes, labels, split=self.split)
        return image, boxes, labels

    def collate_fn(self, batch):
        """
        Each batch can have different number of objects.
        We will pass this collate function to the DataLoader.
        You can define this function outside the class as well.

        :param batch: iterable items from __getitem(), size equal to batch size
        :return: a tensor of images, lists of varying-size tensors of 
                 bounding boxes, labels, and difficulties
        """
        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        # return a tensor (N, 3, 300, 300), 3 lists of N tesnors each
        return images, boxes, labels