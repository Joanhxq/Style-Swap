# -*- coding: utf-8 -*-


from torch import nn

vgg_normalised_conv2_1 = nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=1),
        nn.ReflectionPad2d(1),
        nn.Conv2d(3, 64, 3),
        nn.ReLU(),
        
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 64, 3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 128, 3),
        nn.ReLU(),
        
        )

