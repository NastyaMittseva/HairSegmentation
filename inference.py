import torch
import os
import cv2
import time
import imageio
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from skimage import img_as_ubyte
from network.model import HairMatteNet
from network.utils import dye_hair, denorm

device = 'cuda'
path_input = 'examples/woman.gif'
path_output = 'examples/result_woman.gif'
model_path = 'experiments/start2/models/'
resume_epochs = 99
color = [0, 0, 255]

############ Create generartor ############
class Generator(Dataset):
    def __init__(self, imgs, transform):
        self.imgs = imgs
        self.transform = transform
        
    def __len__(self):
        return len(self.imgs)    
                  
    def __getitem__(self, idx): 
        image = self.transform(Image.fromarray(cv2.cvtColor(self.imgs[idx], cv2.COLOR_BGR2RGB)))   
        return np.asarray(image)

############ Initialize network and load weights ############
net = HairMatteNet()
net.to(device)
net_path =  os.path.join(model_path, '{}_epoch-HairMatteNet.ckpt'.format(resume_epochs))
net.load_state_dict(torch.load(net_path, map_location=lambda storage, loc: storage))  
net.eval()

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

############ Main block ############
start = time.time()
video = imageio.mimread(path_input, memtest=False)
video = [frame[..., :3] for frame in video]
old_size = video[0].shape
batch_size = len(video)
print('Number of frames: ', batch_size)

data = Generator(video, transform = transform)
dataloader = DataLoader(data, shuffle=False, batch_size=batch_size, num_workers=1)

result = []
for i, data in enumerate(dataloader, 0):
    image = data.to(device)
    with torch.no_grad():
        pred = net(image)
    
    for j in range(pred.shape[0]):
        color_image = dye_hair(denorm(image[j:j + 1]), torch.argmax(pred[j:j + 1],1).unsqueeze(0), color, 0.1)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        color_image = cv2.resize(color_image, (old_size[1], old_size[0]))
        result.append(color_image)

imageio.mimsave(path_output, [img_as_ubyte(frame) for frame in result], format='GIF', duration=0.05)  
print('Spend time ', time.time() - start)
