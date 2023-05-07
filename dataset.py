import os
from PIL import Image
from torch.utils.data import Dataset


class CartoonFramesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.transform = transform
        self.images = os.listdir(self.root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        filename = self.images[index]
        path = os.path.join(self.root, filename)
        image = Image.open(path)

        image = self.transform(image)

        return image
