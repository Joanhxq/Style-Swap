# -*- coding: utf-8 -*-


from torch import nn

vgg_normalised_conv1_1 = nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=1),
        nn.ReflectionPad2d(1),
        nn.Conv2d(3, 64, 3),
        nn.ReLU(),

        )

