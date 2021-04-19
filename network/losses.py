import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
    
    
class ImageGradientLoss(_WeightedLoss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(ImageGradientLoss, self).__init__(size_average, reduce, reduction)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, pred, gray_image):
        size = pred.size()
        pred = pred[:, 1, :, :].view(size[0], 1, size[2], size[3]).float()
        gradient_tensor_x = torch.Tensor([[1.0, 0.0, -1.0],
                                          [2.0, 0.0, -2.0],
                                          [1.0, 0.0, -1.0]]).to(self.device).view((1, 1, 3, 3))

        gradient_tensor_y = torch.Tensor([[1.0, 2.0, 1.0],
                                          [0.0, 0.0, 0.0],
                                          [-1.0, -2.0, -1.0]]).to(self.device).view((1, 1, 3, 3))

        I_x = F.conv2d(gray_image, gradient_tensor_x)
        G_x = F.conv2d(pred, gradient_tensor_x)

        I_y = F.conv2d(gray_image, gradient_tensor_y)
        G_y = F.conv2d(pred, gradient_tensor_y)

        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2)+ 1e-6)

        gradient = 1 - torch.pow(I_x * G_x + I_y * G_y, 2)
        gradient = torch.where(gradient > 0, gradient, torch.tensor(0.).to(self.device))
#         gradient = gradient if gradient > 0 else 0
        
        image_gradient_loss = torch.sum(torch.mul(G, gradient)) / (torch.sum(G) + 1e-6)

        return image_gradient_loss


def iou_metric(pred, mask):
    pred = torch.argmax(pred, 1).long()
    mask = torch.squeeze(mask).long()
    Union = torch.where(pred > mask, pred, mask)
    Overlep = torch.mul(pred, mask)
    metric = torch.div(torch.sum(Overlep).float(), torch.sum(Union).float())
    return metric


def acc_metric(pred, mask):
    pred = torch.argmax(pred, 1).long()
    mask = torch.squeeze(mask).long()
    all_ones = torch.ones_like(mask)
    all_zeros = torch.zeros_like(mask)
    Right = torch.where(pred == mask, all_ones, all_zeros)
    metric = torch.div(torch.sum(Right).float(), torch.sum(all_ones).float())
    return metric


def F1_metric(pred, mask):
    pred = torch.argmax(pred, 1).long()
    mask = torch.squeeze(mask).long()
    all_ones = torch.ones_like(mask)
    all_zeros = torch.zeros_like(mask)
    Overlep = torch.mul(pred, mask)
    precision = torch.div(torch.sum(Overlep).float(), torch.sum(pred).float())
    recall = torch.div(torch.sum(Overlep).float(), torch.sum(mask).float())
    F1score = torch.div(torch.mul(precision, recall), torch.add(precision, recall))
    return F1score*2
