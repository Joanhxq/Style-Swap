# -*- coding: utf-8 -*-

import torch
from torch import nn
from models.autoencoder_vgg19.vgg19_1 import vgg_normalised_conv1_1, feature_invertor_conv1_1
from models.autoencoder_vgg19.vgg19_2 import vgg_normalised_conv2_1, feature_invertor_conv2_1
from models.autoencoder_vgg19.vgg19_3 import vgg_normalised_conv3_1, feature_invertor_conv3_1
from models.autoencoder_vgg19.vgg19_4 import vgg_normalised_conv4_1, feature_invertor_conv4_1
from models.autoencoder_vgg19.vgg19_5 import vgg_normalised_conv5_1, feature_invertor_conv5_1


class Encoder(nn.Module):
    
    def __init__(self, relu):
        super(Encoder, self).__init__()
        assert(type(relu).__name__ == 'int' and 0 < relu < 6)
    
        if relu == 1:
            self.model = vgg_normalised_conv1_1.vgg_normalised_conv1_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_1/vgg_normalised_conv1_1.pth"))
            
        elif relu == 2:
            self.model = vgg_normalised_conv2_1.vgg_normalised_conv2_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_2/vgg_normalised_conv2_1.pth"))

        elif relu == 3:
            self.model = vgg_normalised_conv3_1.vgg_normalised_conv3_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_3/vgg_normalised_conv3_1.pth"))

        elif relu == 4:
            self.model = vgg_normalised_conv4_1.vgg_normalised_conv4_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_4/vgg_normalised_conv4_1.pth"))

        else:
            self.model = vgg_normalised_conv5_1.vgg_normalised_conv5_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_5/vgg_normalised_conv5_1.pth"))

    def forward(self, x):
        output = self.model(x)
        return output


class Decoder(nn.Module):
    
    def __init__(self, relu):
        super(Decoder, self).__init__()
        assert(type(relu).__name__ == 'int' and 0 < relu < 6)
        
        if relu == 1:
            self.model = feature_invertor_conv1_1.feature_invertor_conv1_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_1/feature_invertor_conv1_1.pth"))
            
        elif relu == 2:
            self.model = feature_invertor_conv2_1.feature_invertor_conv2_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_2/feature_invertor_conv2_1.pth"))

        elif relu == 3:
            self.model = feature_invertor_conv3_1.feature_invertor_conv3_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_3/feature_invertor_conv3_1.pth"))

        elif relu == 4:
            self.model = feature_invertor_conv4_1.feature_invertor_conv4_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_4/feature_invertor_conv4_1.pth"))

        else:
            self.model = feature_invertor_conv5_1.feature_invertor_conv5_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_5/feature_invertor_conv5_1.pth"))

    def forward(self, x):
        output = self.model(x)
        return output

#def Decoder(relu):
#    model = feature_invertor_conv3_1.feature_invertor_conv3_1()
#    model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_3/feature_invertor_conv3_1_res.pth"))
#    return model

        




