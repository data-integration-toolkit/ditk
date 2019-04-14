import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from modules.misgan_modules.misgan_nn import *
import os
from os.path import join
import numpy as np
import pickle

DATALOADER_PATH = 'data/data_misgan/*_test.data_loader'

def test(fname):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    K, D = 16, 16

    if not os.path.exists('result/misgan_result'):
        os.mkdir('result/misgan_result')
    '''
    load data
    '''

    batch_size = 10
    fpath = DATALOADER_PATH.replace('*', fname)

    if not os.path.exists(fpath):
        print("path not found{0}".format(fpath))
        return
    with open(fpath, 'rb') as f:
        data_loader = pickle.load(f)

    data_samples, mask_samples, _ = next(iter(data_loader))

    nz = 128  # dimensionality of the latent code

    data_gen = ConvDataGenerator(K, D).to(device)
    mask_gen = ConvMaskGenerator(K, D).to(device)

    data_critic = ConvCritic(K, D).to(device)
    mask_critic = ConvCritic(K, D).to(device)

    data_noise = torch.empty(batch_size, nz, device=device)
    mask_noise = torch.empty(batch_size, nz, device=device)

    for real_data, real_mask, _ in data_loader:
        real_data = real_data.to(device)
        real_mask = real_mask.to(device).float()

        # Update discriminators' parameters
        data_noise.normal_()
        mask_noise.normal_()

        fake_data = data_gen(data_noise)
        fake_mask = mask_gen(mask_noise)

        masked_fake_data = mask_data(fake_data, fake_mask)
        masked_real_data = mask_data(real_data, real_mask)

        update_data_critic(masked_real_data, masked_fake_data)
        update_mask_critic(real_mask, fake_mask)

    data_loss = -data_critic(masked_fake_data).mean()
    mask_loss = -mask_critic(fake_mask).mean()


    print("MisGAN data loss = {0}".format(data_loss))
    print("MisGAN mask loss = {0}".format(mask_loss))

    with open('result/misgan_result/test_misgan_loss.txt', 'wb') as f:
        f.write(str(data_loss, mask_loss))

    for real_data, real_mask, index in data_loader:
        # find out if input of any size works too
        real_data = real_data.to(device)
        real_mask = real_mask.to(device).float()

        masked_real_data = mask_data(real_data, real_mask)

        # Update discriminators' parameters
        data_noise.normal_()
        fake_data = data_gen(data_noise)

        mask_noise.normal_()
        fake_mask = mask_gen(mask_noise)
        masked_fake_data = mask_data(fake_data, fake_mask)

        impu_noise.uniform_()
        imputed_data = imputer(real_data, real_mask, impu_noise)

        update_data_critic(masked_real_data, masked_fake_data)
        update_mask_critic(real_mask, fake_mask)
        update_impu_critic(fake_data, imputed_data)

    data_loss = -data_critic(masked_fake_data).mean()
    mask_loss = -mask_critic(fake_mask).mean()
    impu_loss = -impu_critic(imputed_data).mean()
    with open('result/misgan_result/test_imputer_loss.txt', 'wb') as f:
        f.write(str(data_loss, mask_loss, impu_loss))