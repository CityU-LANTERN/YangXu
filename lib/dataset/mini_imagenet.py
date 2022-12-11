import os
import pickle
from PIL import Image

import torch
from torch.utils.data import Dataset


class MiniImageNet(Dataset):

    def __init__(self, root_path, split='train', transform=None):
        self.transform = transform
        split_tag = split
        if split == 'train':
            split_tag = 'train_phase_train'
        split_file = 'miniImageNet_category_split_{}.pickle'.format(split_tag)
        with open(os.path.join(root_path, split_file), 'rb') as f:
            pack = pickle.load(f, encoding='latin1')
        data = pack['data']
        label = pack['labels']
        data = [Image.fromarray(x) for x in data]

        min_label = min(label)
        label = [x - min_label for x in label]
        
        self.data = data
        self.label = label
        self.n_classes = max(self.label) + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img = self.data[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.label[i]


