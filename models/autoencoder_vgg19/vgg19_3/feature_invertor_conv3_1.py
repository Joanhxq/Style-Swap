# -*- coding: utf-8 -*-


from torch import nn

feature_invertor_conv3_1 = nn.Sequential(

        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 128, 3),
        nn.ReLU(),
        
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 128, 3),
        nn.ReLU(),
        
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 64, 3),
        nn.ReLU(),
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 64, 3),
        nn.ReLU(),
        
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 3, 3),
        
        )


#class ResidualBlock(nn.Module):
#    def __init__(self, channel):
#        super(ResidualBlock, self).__init__()
#         
#        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
#        self.relu1 = nn.ReLU()
#        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
#        self.relu2 = nn.ReLU()
#        self.conv3 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
#        self.relu3 = nn.ReLU()
#         
#    def forward(self, x):
#        res = x        
#        out = self.conv1(x)
#        res_1 = res + out
#        out = self.conv2(self.relu1(res_1))
#        res_2 = res_1 + out
#        out = self.relu3(self.conv3(self.relu2(res_2)))
#        return res_2 + out
#     
#class feature_invertor_conv3_1(nn.Module):
#    def __init__(self):
#        super(feature_invertor_conv3_1, self).__init__()
#         
#        self.features = nn.Sequential(
#        ResidualBlock(256),
#        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
#        nn.ReLU(),
#        nn.Upsample(scale_factor=2, mode='nearest'),
#        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#        nn.ReLU(),
#        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
#        nn.ReLU(),
#        nn.Upsample(scale_factor=2, mode='nearest'),
#        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#        nn.ReLU(),
#        nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)                
#        )
#    
#    def forward(self, x):
#        output = self.features(x)
#        return output











