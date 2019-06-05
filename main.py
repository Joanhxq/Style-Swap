# -*- coding: utf-8 -*-


import torch
from torchvision import transforms
import os
import encoder_decoder_factory
from torch.utils.data import DataLoader
from torch import nn
from style_swap import style_swap
from torchvision.utils import save_image
from dataset import FlatFolderDataset
import ipdb


class Config(object):
    contentPath = "./images/content"   # Path of directory containing content images to be transformed
    stylePath = "./images/style"       # Path of directory containing style images to be transformed
    img_hsize = 256                     # Reshape the image to have new size
    img_wsize = 256
    patch_size = 3                     # Patch size of the style swap
    relu_level = 3                     # Style swapping in different layers of VGG19
    max_epoch = 2                      # Numbers of iterations
    minibatch = 2                      # The batch size of each training
    lambda_weight = 1e-5               # The weight of the total variation regularization
    lr = 1e-2                          # The learning rate of Adam
    beta1 = 0.5                        # The parameter of Adam_beta1
    gpu = True                         # Flag to enables GPU to accelerate computations
    out_dir = './outputs'              # Path of the directory store stylized images

opt = Config()

def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_) 
    opt.device = torch.device('cuda') if opt.gpu else torch.device('cpu')
    os.makedirs(opt.out_dir, exist_ok=True)
        
    VggNet = encoder_decoder_factory.Encoder(3).to(device=opt.device)
    InvNet = encoder_decoder_factory.Decoder(3).to(device=opt.device)
    VggNet.train()
    InvNet.train()
    
    for param in InvNet.parameters():
        param.requires_grad = True
    
    transforms_ = transforms.Compose([
            transforms.Resize(size=(opt.img_hsize, opt.img_wsize)),
            transforms.CenterCrop(size=(opt.img_hsize, opt.img_wsize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ])
    
    content_dataset = FlatFolderDataset(root=opt.contentPath, transform=transforms_)
    style_dataset = FlatFolderDataset(root=opt.stylePath, transform=transforms_)
    
    content_dataloader = DataLoader(content_dataset, batch_size=opt.minibatch, shuffle=True, drop_last=True)
    style_dataloader = DataLoader(style_dataset, batch_size=opt.minibatch, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(InvNet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    
    criterion_l2 = nn.MSELoss(reduce=True, size_average=True).to(device=opt.device)
    criterion_tv = nn.MSELoss(reduce=True, size_average=False).to(device=opt.device)
    
    for ii, content in enumerate(content_dataloader):
        content_img = content['img'].to(device=opt.device)
        cf = VggNet(content_img)
        for jj, style in enumerate(style_dataloader):
            for epoch in range(opt.max_epoch):
                style_img = style['img'].to(device=opt.device)  # size: 2 x 3 x 256 x 256
                sf = VggNet(style_img)                          # cf, sf size is 2 x 256 x 64 x 64
                csf = style_swap(cf, sf, opt)                   # csf size is : 2 x 256 x 64 x 64
                I_stylized = InvNet(csf)                        # size: 2 x 3 x 256 x 256
                P_stylized = VggNet(I_stylized)                 # size: 2 x 256 x 64 x 64
                
                input_param = nn.Parameter(I_stylized.data)
                optimizer = torch.optim.Adam([input_param], lr=opt.lr, betas=(opt.beta1, 0.999))
                
                optimizer.zero_grad()
                                
                loss_stylized = criterion_l2(P_stylized, csf.detach())
                
                zero_labels_1 = torch.zeros_like(I_stylized[:,:,1:,:])
                zero_labels_2 = torch.zeros_like(I_stylized[:,:,:,1:])
                loss_tv = criterion_tv(I_stylized[:,:,1:,:]-I_stylized[:,:,:-1,:], zero_labels_1) + criterion_tv(I_stylized[:,:,:,1:]-I_stylized[:,:,:,:-1], zero_labels_2)

                loss = loss_stylized + opt.lambda_weight * loss_tv
                loss.backward()
                
                optimizer.step()
                
                save_image(I_stylized.cpu().detach().squeeze(0), 
                           os.path.join(opt.out_dir, (str(content['img_name'][0]) + '_stylized_by_' + str(style['img_name'][0]) + '_epoch_' + str(epoch) + '.png')), normalize=True, range=(-1,1))
#                ipdb.set_trace()
            print("loss: %.4f\tloss_stylized: %.4f loss_tv: %.4f" % (loss.item()/opt.minibatch, loss_stylized.item()/opt.minibatch, loss_tv.item()/opt.minibatch))
            
    
if __name__ == '__main__':
    import fire
    fire.Fire()







