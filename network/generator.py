import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


class Generator(Dataset):
    def __init__(self, training_data_path, mode, image_size):
        self.training_data = []
        images = os.listdir(training_data_path)
        for image in images:
            self.training_data.append(training_data_path + image) 
        self.mode = mode
        self.image_size = image_size
        
    def __len__(self):
        return len(self.training_data)    
                  
    def __getitem__(self, idx):
        mask = Image.open(self.training_data[idx]).convert('L')
        image = Image.open(self.training_data[idx].replace('_label', '_img').replace('png', 'jpg')).convert('RGB')
        
        image, gray_image, mask = transform(image, mask, self.mode)
        return image, gray_image, mask
    

def transform(image, mask, mode, image_size=224):
    if mode == 'train':
        resized_num = int(random.random() * image_size)
        resize = transforms.Resize(size=(image_size + resized_num, image_size + resized_num))
        image = resize(image)
        mask = resize(mask)
        
        if random.random() > 0.5:
            num_pad = int(random.random() * image_size / 4)
            image = TF.pad(image, num_pad, padding_mode='edge')
            mask = TF.pad(mask, num_pad)
        
        if random.random() > 0.5:
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(image_size, image_size))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() > 0.7:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

    resize = transforms.Resize(size=(image_size, image_size))
    image = resize(image)
    mask = resize(mask)
    
    gray_image = TF.to_grayscale(image)

    image = TF.to_tensor(image)
    mask = TF.to_tensor(mask)
    gray_image = TF.to_tensor(gray_image)

    image = TF.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    return image, gray_image, mask