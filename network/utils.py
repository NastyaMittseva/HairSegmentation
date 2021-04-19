import cv2
import numpy as np
import random

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)
    
    
def dye_hair(image, mask, color, alpha = 0.3):
    image = image[0].data.detach().cpu().numpy() * 255
    image = image.transpose(1, 2, 0).astype(np.uint8)
    
    mask = mask[0][0].data.cpu().numpy() * 255
    mask = mask.astype(np.uint8)

#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     smooth_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)

    color_hair = np.copy(image)
    color_hair[(mask != 0)] = color
    color_hair_w = cv2.addWeighted(color_hair, alpha, image, 1 - alpha, 0)
    return color_hair_w