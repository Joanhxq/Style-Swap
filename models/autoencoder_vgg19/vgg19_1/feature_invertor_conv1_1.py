# -*- coding: utf-8 -*-


from torch import nn

feature_invertor_conv1_1 = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 3, 3),
        
        )


