import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from network.model import *
from network.losses import *
from network.utils import *
from network.generator import Generator
from torchvision.utils import save_image
from torch.optim.adadelta import Adadelta
import tensorboardX 
import torchvision
import time
import datetime
import os
from PIL import Image


class HairSegmentation(object):
    def __init__(self, training_data_path, valid_data_path, test_data_path, resolution, num_classes, decay_epoch, lr, rho, eps, decay, gradient_loss_weight, resume_epochs, log_step, sample_step, num_epochs, batch_size, train_results_dir, valid_results_dir, test_results_dir, model_save_dir, log_dir):
        
        self.training_data_path = training_data_path
        self.valid_data_path = valid_data_path
        self.test_data_path = test_data_path
        self.resolution = resolution
        self.num_classes = num_classes
        
        self.decay_epoch = decay_epoch
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.decay = decay
        self.gradient_loss_weight = gradient_loss_weight
        
        self.resume_epochs = resume_epochs
        self.log_step = log_step
        self.sample_step = sample_step
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        self.train_results_dir = train_results_dir
        self.valid_results_dir = valid_results_dir
        self.test_results_dir = test_results_dir
        self.model_save_dir = model_save_dir
        self.log_dir = log_dir
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.colors = [(0, 0, 255), (255, 0, 0), (255, 0, 255), (255, 166, 0), (255, 255, 0), (0, 255, 0), (0, 191, 255), (255, 192, 203)]
        
        self.create_generator()
        self.build_model()
        self.writer = tensorboardX.SummaryWriter(self.log_dir) 
        
        
    def create_generator(self): 
        self.transform = transforms.Compose([transforms.Resize((self.resolution,self.resolution)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_data = Generator(self.training_data_path, 'train', self.resolution)
        self.train_dataloader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size, num_workers=4, drop_last=True)
        
        valid_data = Generator(self.valid_data_path, 'valid', self.resolution)
        self.valid_dataloader = DataLoader(valid_data, shuffle=True, batch_size=self.batch_size, num_workers=4, drop_last=True)
        
        test_data = Generator(self.test_data_path, 'test', self.resolution)
        self.test_dataloader = DataLoader(test_data, shuffle=True, batch_size=self.batch_size, num_workers=4, drop_last=True)
        
        
    def build_model(self):
        self.net = HairMatteNet()
        self.net.to(self.device)
        self.optimizer = Adadelta(self.net.parameters(), lr=self.lr, eps=self.eps, rho=self.rho, weight_decay=self.decay)        
        
        
    def restore_model(self, resume_epochs):
        print('Loading the trained models from epoch {}...'.format(resume_epochs))
        net_path =  os.path.join(self.model_save_dir, '{}_epoch-HairMatteNet.ckpt'.format(resume_epochs))
        self.net.load_state_dict(torch.load(net_path, map_location=lambda storage, loc: storage))
    
    
    def train_epoch(self, epoch, start_time):
        self.net.train()
        for i, data in enumerate(self.train_dataloader, 0):
            image = data[0].to(self.device)
            gray_image = data[1].to(self.device)
            mask = data[2].to(self.device)

            pred = self.net(image)

            pred_flat = pred.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
            mask_flat = mask.squeeze(1).view(-1).long()

            image_gradient_loss = self.image_gradient_criterion(pred, gray_image)
            bce_loss = self.bce_criterion(pred_flat, mask_flat)
            loss = bce_loss + self.gradient_loss_weight * image_gradient_loss 

            iou = iou_metric(pred, mask)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {}
            losses['train_bce_loss'] = bce_loss.item()
            losses['train_image_gradient_loss'] = self.gradient_loss_weight * image_gradient_loss
            losses['train_loss'] = loss
            losses['train_iou'] = iou

            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}], Epoch [{}/{}]".format(et, i + 1, len(self.train_dataloader), epoch, self.num_epochs)
                for tag, value in losses.items():
                    log += ", {}: {:.4f}".format(tag, value)
                    self.writer.add_scalar(tag, value, epoch*len(self.train_dataloader) + i + 1)
                print(log)

            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    out_results = []
                    for j in range(10):
                        out_results.append(denorm(image[j:j + 1]).data.cpu())
                        out_results.append(mask.expand(-1, 3, -1, -1)[j:j + 1].data.cpu())
                        out_results.append(torch.argmax(pred[j:j + 1],1).unsqueeze(0).expand(-1, 3, -1, -1).data.cpu()) 

                        color = random.choice(self.colors)

                        result = dye_hair(denorm(image[j:j + 1]), mask[j:j + 1], color)
                        result = self.transform(Image.fromarray(result)).unsqueeze(0)
                        out_results.append(denorm(result))

                        result = dye_hair(denorm(image[j:j + 1]), torch.argmax(pred[j:j + 1],1).unsqueeze(0), color)
                        result = self.transform(Image.fromarray(result)).unsqueeze(0)
                        out_results.append(denorm(result))    

                    results_concat = torch.cat(out_results)
                    results_path = os.path.join(self.train_results_dir, '{}_epoch_train_results.jpg'.format(epoch))
                    save_image(results_concat, results_path, nrow=5, padding=0)
                    print('Saved real and fake images into {}...'.format(results_path))

        if (epoch+1) % 2 == 0:
            net_path =  os.path.join(self.model_save_dir, '{}_epoch-HairMatteNet.ckpt'.format(epoch))
            torch.save(self.net.state_dict(), net_path)
            print('Saved model checkpoints into {}...'.format(self.model_save_dir))
    
    
    def valid_epoch(self, epoch):
        self.net.eval()
        losses = {'valid_bce_loss': 0, 'valid_image_gradient_loss': 0, 'valid_loss': 0, 'valid_iou': 0}
        for i, data in enumerate(self.valid_dataloader, 0):
            image = data[0].to(self.device)
            gray_image = data[1].to(self.device)
            mask = data[2].to(self.device)
            
            with torch.no_grad():
                pred = self.net(image)

            pred_flat = pred.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
            mask_flat = mask.squeeze(1).view(-1).long()

            image_gradient_loss = self.image_gradient_criterion(pred, gray_image)
            bce_loss = self.bce_criterion(pred_flat, mask_flat)
            loss = bce_loss + self.gradient_loss_weight * image_gradient_loss 

            iou = iou_metric(pred, mask)

            losses['valid_bce_loss'] += bce_loss.item()
            losses['valid_image_gradient_loss'] += self.gradient_loss_weight * image_gradient_loss
            losses['valid_loss'] += loss
            losses['valid_iou'] += iou
            
            if i == 0:
                with torch.no_grad():
                    out_results = []
                    for j in range(10):
                        out_results.append(denorm(image[j:j + 1]).data.cpu())
                        out_results.append(mask.expand(-1, 3, -1, -1)[j:j + 1].data.cpu())
                        out_results.append(torch.argmax(pred[j:j + 1],1).unsqueeze(0).expand(-1, 3, -1, -1).data.cpu()) 

                        color = random.choice(self.colors)

                        result = dye_hair(denorm(image[j:j + 1]), mask[j:j + 1], color)
                        result = self.transform(Image.fromarray(result)).unsqueeze(0)
                        out_results.append(denorm(result))

                        result = dye_hair(denorm(image[j:j + 1]), torch.argmax(pred[j:j + 1],1).unsqueeze(0), color)
                        result = self.transform(Image.fromarray(result)).unsqueeze(0)
                        out_results.append(denorm(result))    

                    results_concat = torch.cat(out_results)
                    results_path = os.path.join(self.valid_results_dir, '{}_epoch_valid_results.jpg'.format(epoch))
                    save_image(results_concat, results_path, nrow=5, padding=0)
                    print('Saved real and fake images into {}...'.format(results_path))
            
        losses['valid_bce_loss'] /= i
        losses['valid_image_gradient_loss'] /= i
        losses['valid_loss'] /= i
        losses['valid_iou'] /= i
            
        log = "Eval ========================= Epoch [{}/{}]".format(epoch, self.num_epochs)
        for tag, value in losses.items():
            log += ", {}: {:.4f}".format(tag, value)
            self.writer.add_scalar(tag, value, epoch*len(self.train_dataloader) + len(self.train_dataloader) + 1)
        print(log)
      
            
    def train(self):
        if self.resume_epochs != 0:
            self.restore_model(self.resume_epochs)
            self.resume_epochs += 1
            
        self.image_gradient_criterion = ImageGradientLoss().to(self.device)
        self.bce_criterion = nn.CrossEntropyLoss().to(self.device)

        start_time = time.time()
        for epoch in range(self.resume_epochs, self.num_epochs, 1):
            self.train_epoch(epoch, start_time)
            self.valid_epoch(epoch)
                
        self.writer.close()
        
        
    def test(self):
        self.restore_model(self.resume_epochs)
        self.net.eval()
        
        iou = 0
        for i, data in enumerate(self.valid_dataloader, 0):
            image = data[0].to(self.device)
            mask = data[2].to(self.device)
            
            with torch.no_grad():
                pred = self.net(image)
                
            iou += iou_metric(pred, mask)
            
            if i == 0:
                with torch.no_grad():
                    out_results = []
                    for j in range(10):
                        out_results.append(denorm(image[j:j + 1]).data.cpu())
                        out_results.append(mask.expand(-1, 3, -1, -1)[j:j + 1].data.cpu())
                        out_results.append(torch.argmax(pred[j:j + 1],1).unsqueeze(0).expand(-1, 3, -1, -1).data.cpu()) 

                        color = random.choice(self.colors)

                        result = dye_hair(denorm(image[j:j + 1]), mask[j:j + 1], color)
                        result = self.transform(Image.fromarray(result)).unsqueeze(0)
                        out_results.append(denorm(result))

                        result = dye_hair(denorm(image[j:j + 1]), torch.argmax(pred[j:j + 1],1).unsqueeze(0), color)
                        result = self.transform(Image.fromarray(result)).unsqueeze(0)
                        out_results.append(denorm(result))    

                    results_concat = torch.cat(out_results)
                    results_path = os.path.join(self.test_results_dir, '{}_epoch_test_results.jpg'.format(self.resume_epochs))
                    save_image(results_concat, results_path, nrow=5, padding=0)
                    print('Saved real and fake images into {}...'.format(results_path))
                    
        iou /= i
        print('Average iou = ', iou.item())