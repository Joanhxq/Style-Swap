# -*- coding: utf-8 -*-


from torch import nn

feature_invertor_conv2_1 = nn.Sequential(
      
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


