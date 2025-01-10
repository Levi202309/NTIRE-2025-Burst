"""
We refer the code made from
https://github.com/z-bingo/kernel-prediction-networks-PyTorch/blob/master/train_eval_syn.py
"""


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from fvcore.nn import FlopCountAnalysis, flop_count_table
import os, sys, time, shutil

from PIL import Image
from torchvision.transforms import transforms
to_pil_image = transforms.ToPILImage()

from DataLoader.custom_data_class import CustomDataset
from models.unet_model import UNet
import pdb

from utils.utils import *
from utils.checkpoint import *



def eval(cuda, mGPU=True):
    print('Eval Process......')

    checkpoint_dir = './checkpoint_dir'
    if not os.path.exists(checkpoint_dir) or len(os.listdir(checkpoint_dir)) == 0:
        print('There is no any checkpoint file in path:{}'.format(checkpoint_dir))
    # the path for saving eval images
    eval_dir = './eval_dir'
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)

    # dataset and dataloader
    data_set = CustomDataset(root_dir="../datasets/val/", transform=transforms.ToTensor(), train=False)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False)
    print("Length of the data_loader :", len(data_loader))



    """
        Your model will be loaded here, via submitted pytorch code and trained parameters.
        You may upload zip file containing my_network.py and my_parameters.pth.tar to the server. 
        The specific guideline for how to submit your model will be provided later. 
    """
    ## your model here. ####
    model = UNet(in_channels=9,  # 9 frames concat through channel dimension
        n_classes=3,        # out channels (RGB)
        depth=4,
        wf=6,
        padding=True,
        batch_norm=False,
        up_mode='upconv')
    
    if cuda:
        model = model.cuda()

    if mGPU:
        model = nn.DataParallel(model)

    # load trained model parameters
    # model.load_state_dict(torch.load('./submission/my_parameters.pth', weights_only=True))
    checkpoint = load_checkpoint(checkpoint_dir, 'best')
    start_epoch = checkpoint['epoch']
    global_step = checkpoint['global_iter']
    best_loss = checkpoint['best_loss']
    model.load_state_dict(checkpoint['state_dict'])

    print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
    print('The model has been completely loaded from the user submission.')

    # parameters and flops
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    flops = FlopCountAnalysis(model, torch.ones(1, 9, 2000, 3000).to(device))
    print(flop_count_table(flops))

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total # of model parameters : {num_params / 1000 / 1000 :.3f}(M)")
    print(f"Total FLOPs of the model : {flops.total() / (1000**4) :.3f}(T)")
    print('\n-------Evaluation started -------\n')


    # switch the eval mode
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        psnr = 0.0
        ssim = 0.0

        for i, (burst_noise, gt) in enumerate(data_loader):
            t0 = time.time()
            if cuda:
                burst_noise = burst_noise.cuda()
                gt = gt.cuda()

            burst_noise = burst_noise.squeeze(2)
            pred = model(burst_noise)

            pred = torch.clamp(pred, 0.0, 1.0)
            t1 = time.time()

            if cuda:
                pred = pred.cpu()
                gt = gt.cpu()
                burst_noise = burst_noise.cpu()
            print('{}-th image is completed.\t| time: {:.2f} seconds.'.format(i,t1 - t0))
            if i < 20:
                for frame in range(9):
                    pil_image = to_pil_image(burst_noise[0][frame])
                    pil_image.save(eval_dir+f'/Scene{i}_input{frame}.png')
                pil_image = to_pil_image(pred[0])
                pil_image.save(eval_dir+f'/Scene{i}_output.png')
            else:
                break
    end_time = time.time()
    print('All images are OK, average PSNR: {:.2f}dB, SSIM: {:.4f}'.format(psnr/(i+1), ssim/(i+1)))
    print(f'Total Validation time : {end_time - start_time} seconds.')

if __name__ == '__main__':
    eval(cuda=True, mGPU=2)