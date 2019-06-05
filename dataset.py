# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import os
from PIL import Image


class FlatFolderDataset(Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform
        
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        img_name = path.split('.')[0]
        return {'img': img, 'img_name': img_name}
    
    def __len__(self):
        return len(self.paths)
