import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image


class MyDataLoader(Dataset):

    def __init__(self, room_flist, input_flist, mask_flist):
        self.room_data = np.genfromtxt(room_flist, dtype=np.str, encoding='utf-8')
        self.input_data = np.genfromtxt(input_flist, dtype=np.str, encoding='utf-8')
        self.mask_data = np.genfromtxt(mask_flist, dtype=np.str, encoding='utf-8')

    def __len__(self):
        return len(self.room_data)

    def __getitem__(self, index):
        room_image = Image.open(self.room_data[index])
        input_image = Image.open(self.input_data[index])
        mask_image = Image.open(self.mask_data[index])

        img_to_tensor = transforms.ToTensor()
        room_image = img_to_tensor(room_image)
        input_image = img_to_tensor(input_image)
        mask_image = img_to_tensor(mask_image)

        return room_image, input_image, mask_image
