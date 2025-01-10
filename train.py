"""
We refer the code made from
https://github.com/z-bingo/kernel-prediction-networks-PyTorch/blob/master/train_eval_syn.py
"""



import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import argparse

import os, sys, time, shutil

from PIL import Image
from torchvision.transforms import transforms
to_pil_image = transforms.ToPILImage()

from DataLoader.custom_data_class import CustomDataset
from models.unet_model import UNet
import pdb

from utils.utils import *
from utils.checkpoint import *

def train(num_threads, cuda, restart_train, mGPU):
    torch.set_num_threads(num_threads)

    batch_size = 2
    lr_decay = 0.9
    lr = 2e-4

    n_epoch = 500

    # checkpoint path
    checkpoint_dir = 'checkpoint_dir'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # logs path
    logs_dir = 'logs_dir'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    shutil.rmtree(logs_dir)

    # dataset and dataloader
    data_set = CustomDataset(root_dir="../datasets/trn/", transform=transforms.ToTensor(), train=True)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False)
    print("Length of the data_loader :", len(data_loader))
    # model here
    model = UNet(in_channels=9,  # 9 frames considered as channel dimension
        n_classes=3,        # out channels (RGB)
        depth=4,
        wf=6,
        padding=True,
        batch_norm=False,
        up_mode='upconv')

    print('\n-------Training started -------\n')

    if cuda:
        model = model.cuda()

    if mGPU:
        model = nn.DataParallel(model)
    model.train()


    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=lr_decay)

    average_loss = MovingAverage(200)
    if not restart_train:
        try:
            checkpoint = load_checkpoint(checkpoint_dir, 'best')
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_iter']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
        except:
            start_epoch = 0
            global_step = 0
            best_loss = np.inf
            print('=> no checkpoint file to be loaded.')
    else:
        start_epoch = 0
        global_step = 0
        best_loss = np.inf
        if os.path.exists(checkpoint_dir):
            pass
        else:
            os.mkdir(checkpoint_dir)
        print('=> training')

    MSE_loss = nn.MSELoss()
    

    for epoch in range(start_epoch, n_epoch):
        epoch_start_time = time.time()
        print('='*20, 'lr={}'.format([param['lr'] for param in optimizer.param_groups]), '='*20)
        
        for step, (burst_noise, gt) in enumerate(data_loader):
            t0 = time.time()
            if cuda:
                burst_noise = burst_noise.cuda()
                gt = gt.cuda()
            burst_noise = burst_noise.squeeze(2)
            pred = model(burst_noise)
            loss = MSE_loss(pred, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            average_loss.update(loss)
            # pdb.set_trace()

            psnr = calculate_psnr(pred.unsqueeze(1), gt.unsqueeze(1))
            ssim = calculate_ssim(pred.unsqueeze(1), gt.unsqueeze(1))
            t1 = time.time()
            # print
            print('{:-4d}\t| epoch {:2d}\t| step {:4d}\t|'
                  ' loss: {:.4f}\t| PSNR: {:.2f}dB\t| SSIM: {:.4f}\t| time:{:.2f} seconds.'
                  .format(global_step, epoch, step, loss, psnr, ssim, t1-t0))
            global_step += 1
            # save images
            if step <20 :
                for frame in range(9):
                    pil_image = to_pil_image(burst_noise[0][frame])
                    pil_image.save(f'./output/Batch{step}_input{frame}.png')
                pil_image = to_pil_image(gt[0])
                pil_image.save(f'./output/Batch{step}_gt.png')
                pil_image = to_pil_image(pred[0])
                pil_image.save(f'./output/Batch{step}_output_E{epoch}.png')
            else:
                break

        print('Epoch {} is finished, time elapsed {:.2f} seconds.'.format(epoch, time.time() - epoch_start_time))
        if epoch % 5 == 0:
            if average_loss.get_value() < best_loss:
                is_best = True
                best_loss = average_loss.get_value()
            else:
                is_best = False

            save_dict = {
                'epoch': epoch,
                'global_iter': global_step,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict()
            }
            save_checkpoint(
                save_dict, is_best, checkpoint_dir, global_step, max_keep=5
            )

        # decay the learning rate
        lr_cur = [param['lr'] for param in optimizer.param_groups]
        if lr_cur[0] > 5e-6:
            scheduler.step()
        else:
            for param in optimizer.param_groups:
                param['lr'] = 5e-6



if __name__ == '__main__':
    train(num_threads=1, cuda=True, restart_train=False, mGPU=2)
