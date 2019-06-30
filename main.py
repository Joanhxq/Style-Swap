# -*- coding: utf-8 -*-


import torch
from torchvision import transforms
import os
from torch.utils.data import DataLoader
from torch import nn
from style_swap import style_swap
from torchvision.utils import save_image
from dataset import PreprocessDataset
from PIL import Image
from model import *
import matplotlib.pyplot as plt
import visdom
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Config(object):
    content = "./images/content"       # Path of directory containing content images to be transformed
    style = "./images/style"           # Path of directory containing style images to be transformed
    img_size = 256                     # Reshape the image to have new size
    patch_size = 3                     # Patch size of the style swap
    relu_level = 3                     # Style swapping in different layers of VGG19
    max_epoch = 3                      # Numbers of iterations
    minibatch = 4                      # The batch size of each training
    tv_weight = 1e-6                   # The weight of the total variation regularization
    lr = 1e-3                          # The learning rate of Adam
    gpu = True                         # Flag to enables GPU to accelerate computations
    out_dir = './outputs'              # Path of the directory to store stylized images
    save_dir = './save_models'         # Path of the directory to store models
    vis = True                         # Use visdiom

opt = Config()


def TVLoss(img, tv_weight):
    w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
    h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss


def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_) 
        
    device = torch.device('cuda') if opt.gpu else torch.device('cpu')
    os.makedirs(opt.out_dir, exist_ok=True)
    os.makedirs(opt.save_dir, exist_ok=True)
    
    if opt.vis:
        vis = visdom.Visdom(env='Style-Swap')
        
    VggNet = VGGEncoder(opt.relu_level).to(device)
    InvNet = Decoder(opt.relu_level).to(device)
    VggNet.train()
    InvNet.train()
        
    train_trans = transforms.Compose([
            transforms.Resize(size=opt.img_size),
            transforms.CenterCrop(size=opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
    
    train_dataset = PreprocessDataset(opt.content, opt.style, train_trans)
    
    train_dataloader = DataLoader(train_dataset, batch_size=opt.minibatch, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(InvNet.parameters(), lr=opt.lr)
    
    criterion = nn.MSELoss()
    
    loss_list = []
    i = 0
    for epoch in range(1, opt.max_epoch+1):
        for _, image in enumerate(train_dataloader):
            content = image['c_img'].to(device)
            style = image['s_img'].to(device)
            cf = VggNet(content)
            sf = VggNet(style)
            csf = style_swap(cf, sf, opt.patch_size, stride=3)
            I_stylized = InvNet(csf)
            I_c = InvNet(cf)
            I_s = InvNet(sf)
            
            P_stylized = VggNet(I_stylized)     # size: 2 x 256 x 64 x 64
            P_c = VggNet(I_c)
            P_s = VggNet(I_s)
            
            loss_stylized = criterion(P_stylized, csf) + criterion(P_c, cf) + criterion(P_s, sf)
            loss_tv = TVLoss(I_stylized, opt.tv_weight)
            loss = loss_stylized + loss_tv
            loss_list.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("%d / %d epoch\tloss: %.4f\tloss_stylized: %.4f loss_tv: %.4f" % (epoch, opt.max_epoch, loss.item()/opt.minibatch, loss_stylized.item()/opt.minibatch, loss_tv.item()/opt.minibatch))
            i += 1
            vis.line(Y=np.array([loss.item()]), X=np.array([i]), win='train_loss', update='append')
        torch.save(InvNet.state_dict(), f'{opt.save_dir}/InvNet_{epoch}_epoch.pth')

    with open('loss_log.txt', 'w') as f:
        for l in loss_list:
            f.write(f'{l}\n')

    
def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res
    
def test(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    device = torch.device('cuda') if opt.gpu else torch.device('cpu')
    os.makedirs(opt.out_dir, exist_ok=True)
        
    VggNet = VGGEncoder(opt.relu_level).to(device)
    InvNet = Decoder(opt.relu_level).to(device)
    InvNet.load_state_dict(torch.load(f'{opt.save_dir}/InvNet_3_epoch.pth'))
    VggNet.train()
    InvNet.train()
    
    content = Image.open(opt.content)
    style = Image.open(opt.style)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])

    content = transform(content).unsqueeze(0).to(device)
    style = transform(style).unsqueeze(0).to(device)

    with torch.no_grad():
        cf = VggNet(content)
        sf = VggNet(style)
        csf = style_swap(cf, sf, opt.patch_size, 3)
        I_stylized = InvNet(csf)   
        I_stylized = denorm(I_stylized, device)

        save_image(I_stylized.cpu(), 
                   os.path.join(opt.out_dir, (opt.content.split('/')[-1].split('.')[0] + '_stylized_by_' + opt.style.split('/')[-1])))


if __name__ == '__main__':
    import fire
    fire.Fire()
