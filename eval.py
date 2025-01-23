"""
We refer the code made from
https://github.com/z-bingo/kernel-prediction-networks-PyTorch/blob/master/train_eval_syn.py
"""


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
import numpy as np
from fvcore.nn import FlopCountAnalysis, flop_count_table
import os, time

from torchvision.transforms import transforms
from DataLoader.custom_data_class import CustomDataset
from models.unet_model import UNet

from utils.utils import *
from utils.checkpoint import *



def eval(cuda, mGPU=True):
    print('Eval Process......')

    checkpoint_dir = './checkpoint_dir'
    if not os.path.exists(checkpoint_dir) or len(os.listdir(checkpoint_dir)) == 0:
        print('There is no any checkpoint file in path:{}'.format(checkpoint_dir))
    # the path for saving eval images
    eval_dir = './res'
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
    flops = FlopCountAnalysis(model, torch.ones(1, 9, 768, 1536).to(device))
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
            
            psnr_t = calculate_psnr(pred.unsqueeze(1), gt.unsqueeze(1))
            ssim_t = calculate_ssim(pred.unsqueeze(1), gt.unsqueeze(1))
            # ssim_t=psnr_t
            psnr += psnr_t
            ssim += ssim_t
            pred = torch.clamp(pred, 0.0, 1.0)
            t1 = time.time()

            if cuda:
                pred = pred.cpu()
                gt = gt.cpu()
                burst_noise = burst_noise.cpu()
            print('{}-th image is completed.\t| PSNR: {:.2f}dB\t| SSIM: {:.4f}\t| time: {:.2f} seconds.'.format(i, psnr_t, ssim_t, t1 - t0))

            # to save the output image
            names = os.listdir("../datasets/val/")
            ii = i * 10

            cv2.imwrite(eval_dir + f'/' + names[ii][:-6] + f'out.tif', (pred[0]*255).permute(1,2,0).cpu().numpy().astype(np.uint8))
            cv2.imwrite(eval_dir + f'/' + names[ii][:-6] + f'gt.tif', (gt[0]*255).permute(1,2,0).cpu().numpy().astype(np.uint8))
            
            
            for frame in range(9):
                # save input frames
                cv2.imwrite(eval_dir + f'/' + names[ii][:-6] + f'input{frame}.tif', (burst_noise[0][frame] * 255).cpu().numpy().astype(np.uint8))

            
    end_time = time.time()
    print('All images are OK, average PSNR: {:.2f}dB, SSIM: {:.4f}'.format(psnr/(i+1), ssim/(i+1)))
    print(f'Total Validation time : {end_time - start_time : .2f} seconds.')

if __name__ == '__main__':
    eval(cuda=True, mGPU=2)
