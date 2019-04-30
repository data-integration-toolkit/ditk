import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from misgan.modules.misgan_nn import *
import os
from os.path import join
import numpy as np
import pickle
import unittest
from time import gmtime, strftime

DATALOADER_PATH = 'data/*.data_loader'
DATA_CRITIC_CHECKPOINT_PATH = 'checkpoint/{0}.csv_train_data_critic.pth'
DATA_GEN_CHECKPOINT_PATH = 'checkpoint/{0}.csv_train_data_gen.pth'
IMPUTER_CRITIC_CHECKPOINT_PATH = 'checkpoint/{0}.csv_train_impute_critic.pth'
IMPUTER_CHECKPOINT_PATH = 'checkpoint/{0}.csv_train_imputer.pth'
MASK_CRITIC_CHECKPOINT_PATH = 'checkpoint/{0}.csv_train_mask_critic.pth'
MASK_GEN_CHECKPOINT_PATH = 'checkpoint/{0}.csv_train_mask_gen.pth'

def test(model, fname):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    t = strftime("%d_%b_%Y_%H_%M_%S_", gmtime())

    K, D = 16, 16

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
    data_gen.load_state_dict(torch.load(DATA_GEN_CHECKPOINT_PATH.format(model)))
    mask_gen.load_state_dict(torch.load(MASK_GEN_CHECKPOINT_PATH.format(model)))
    data_gen.eval()
    mask_gen.eval()

    data_critic = ConvCritic(K, D).to(device)
    mask_critic = ConvCritic(K, D).to(device)
    data_critic.load_state_dict(torch.load(DATA_CRITIC_CHECKPOINT_PATH.format(model)))
    mask_critic.load_state_dict(torch.load(MASK_CRITIC_CHECKPOINT_PATH.format(model)))
    data_critic.eval()
    mask_critic.eval()

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

    data_loss = -data_critic(masked_fake_data).mean()
    mask_loss = -mask_critic(fake_mask).mean()

    print("MisGAN data loss = {0}".format(data_loss.item()))
    print("MisGAN mask loss = {0}".format(mask_loss.item()))

    with open('result/{0}test_misgan_loss.txt'.format(t), 'w') as f:
        f.writelines("Data_loss: {0} mask_loss: {1}".format(data_loss.item(), mask_loss.item()))

    imputer = Imputer(K, D).to(device)
    imputer.load_state_dict(torch.load(IMPUTER_CHECKPOINT_PATH.format(model)))
    imputer.eval()

    impu_critic = ConvCritic(K, D).to(device)
    impu_critic.load_state_dict(torch.load(IMPUTER_CRITIC_CHECKPOINT_PATH.format(model)))
    impu_critic.eval()

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

        imputed_data = imputer.inference(real_data, real_mask)

    data_loss = -data_critic(masked_fake_data).mean()
    mask_loss = -mask_critic(fake_mask).mean()
    impu_loss = -impu_critic(imputed_data).mean()
    with open('result/{0}test_imputer_loss.txt'.format(t), 'w') as f:
        f.writelines("Data_loss: {0} mask_loss: {1} impute_loss: {2}".format(data_loss.item(), mask_loss.item(), impu_loss.item()))