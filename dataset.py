import glob
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class PreprocessDataset(Dataset):
    def __init__(self, content_dir, style_dir, train_trans):
        content_images = glob.glob((content_dir + '/*'))
        np.random.shuffle(content_images)
        style_images = glob.glob((style_dir + '/*'))
        np.random.shuffle(style_images)
        self.images_pairs = list(zip(content_images, style_images))
        self.transforms = train_trans

    def __len__(self):
        return len(self.images_pairs)

    def __getitem__(self, index):
        content_name, style_name = self.images_pairs[index]
        content_image = Image.open(content_name).convert('RGB')
        style_image = Image.open(style_name).convert('RGB')
        if self.transforms:
            content_image = self.transforms(content_image)
            style_image = self.transforms(style_image)
        return {'c_img': content_image, 'c_name': content_name, 's_img': style_image, 's_name': style_name}
