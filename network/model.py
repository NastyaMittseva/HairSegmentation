import torch.nn as nn
from network.components import *


class Encoder(nn.Module):
    def __init__(self, kernel_size=3):
        super(Encoder, self).__init__()
        
        self.encode_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=kernel_size, stride=2, padding=1),
            Layer_Depwise_Encode(32, 64, reserve=True)
        )
        self.encode_layer2 = nn.Sequential(
            Layer_Depwise_Encode(64, 128),
            Layer_Depwise_Encode(128, 128),
        )
        self.encode_layer3 = nn.Sequential(
            Layer_Depwise_Encode(128, 256),
            Layer_Depwise_Encode(256,256)
        )
        self.encode_layer4 = nn.Sequential(
            Layer_Depwise_Encode(256, 512),
            Layer_Depwise_Encode(512, 512),
            Layer_Depwise_Encode(512, 512),
            Layer_Depwise_Encode(512, 512),
            Layer_Depwise_Encode(512, 512),
            Layer_Depwise_Encode(512, 512),
        )
        self.encode_layer5 = nn.Sequential(
            Layer_Depwise_Encode(512, 1024),
            Layer_Depwise_Encode(1024, 1024)
        )
        
        self.encode_to_decoder4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1)
        self.encode_to_decoder3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)
        self.encode_to_decoder2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.encode_to_decoder1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1) 
            
    def forward(self, x):
        encode_layer1 = self.encode_layer1(x)
        encode_layer2 = self.encode_layer2(encode_layer1)
        encode_layer3 = self.encode_layer3(encode_layer2)
        encode_layer4 = self.encode_layer4(encode_layer3)
        encode_layer5 = self.encode_layer5(encode_layer4)

        encode_layer4 = self.encode_to_decoder4(encode_layer4)
        encode_layer3 = self.encode_to_decoder3(encode_layer3)
        encode_layer2 = self.encode_to_decoder2(encode_layer2)
        encode_layer1 = self.encode_to_decoder1(encode_layer1)
        
        return [encode_layer5, encode_layer4, encode_layer3, encode_layer2, encode_layer1]     
    
    
class Decoder(nn.Module):
    def __init__(self, kernel_size=3):
        super(Decoder, self).__init__()
        
        self.decode_layer1 = nn.Upsample(scale_factor=2)
        self.decode_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=64, kernel_size=1),
            Layer_Depwise_Decode(in_channel=64, out_channel=64, kernel_size=kernel_size),
            nn.Upsample(scale_factor=2)
        )
        self.decode_layer3 = nn.Sequential(
            Layer_Depwise_Decode(in_channel=64, out_channel=64, kernel_size=kernel_size),
            nn.Upsample(scale_factor=2)
        )
        self.decode_layer4 = nn.Sequential(
            Layer_Depwise_Decode(in_channel=64, out_channel=64, kernel_size=kernel_size),
            nn.Upsample(scale_factor=2)
        )
        self.decode_layer5 = nn.Sequential(
            Layer_Depwise_Decode(in_channel=64, out_channel=64, kernel_size=kernel_size),
            nn.Upsample(scale_factor=2),
            Layer_Depwise_Decode(in_channel=64, out_channel=64, kernel_size=kernel_size),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=kernel_size, padding=1)
        )

        self.softmax = nn.Softmax(dim=1)
            
    def forward(self, features):
        encode_layer5, encode_layer4, encode_layer3, encode_layer2, encode_layer1 = features
        decode_layer1 = self.decode_layer1(encode_layer5) + encode_layer4
        decode_layer2 = self.decode_layer2(decode_layer1) + encode_layer3
        decode_layer3 = self.decode_layer3(decode_layer2) + encode_layer2
        decode_layer4 = self.decode_layer4(decode_layer3) + encode_layer1
        decode_layer5 = self.decode_layer5(decode_layer4)

        out = self.softmax(decode_layer5)
        return out
        

class HairMatteNet(nn.Module):
    def __init__(self, ):
        super(HairMatteNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self._init_weight()
        
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)