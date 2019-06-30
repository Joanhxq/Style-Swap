import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


class VGGEncoder(nn.Module):
    def __init__(self, relu_level):
        super(VGGEncoder, self).__init__()
        assert(type(relu_level).__name__ == 'int' and 0 < relu_level < 6)
        vgg = vgg19(pretrained=True).features
        if relu_level == 1:
            self.model = vgg[:2]
        elif relu_level == 2:
            self.model = vgg[:7]
        elif relu_level == 3:
            self.model = vgg[:12]
        elif relu_level == 4:
            self.model = vgg[:21]
        else:
            self.model = vgg[:30]
        
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images):
        output = self.model(images)
        return output


decoder = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3, 1),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3, 1),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3, 1),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3, 1),
        nn.ReLU(),    # :13
        
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 256, 3, 1),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1),
        nn.ReLU(),    # :25
        
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 128, 3, 1),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 128, 3, 1),
        nn.ReLU(),    # :32
        
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 64, 3, 1),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 64, 3, 1),
        nn.ReLU(),    # :39
        
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 3, 3, 1),
    
        )


class Decoder(nn.Module):
    def __init__(self, relu_level, decoder=decoder):
        super().__init__()
        decoder = list(decoder.children())
        if relu_level == 1:
            self.model = nn.Sequential(*decoder[39:])
        elif relu_level == 2:
            self.model = nn.Sequential(*decoder[32:])
        elif relu_level == 3:
            self.model = nn.Sequential(*decoder[25:])
        elif relu_level== 4:
            self.model = nn.Sequential(*decoder[13:])
        elif self.relu_level == 5:
            self.model = nn.Sequential(*decoder)
            
    def forward(self, features):
        output = self.model(features)
        return output
