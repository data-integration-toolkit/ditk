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

DATALOADER_PATH = 'data/data_misgan/*_train.data_loader'

def train(fname):
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

    '''
    MisGAN Initialization
    '''

    nz = 128  # dimensionality of the latent code
    n_critic = 5
    alpha = .2

    data_gen = ConvDataGenerator(K, D).to(device)
    mask_gen = ConvMaskGenerator(K, D).to(device)

    data_critic = ConvCritic(K, D).to(device)
    mask_critic = ConvCritic(K, D).to(device)

    data_noise = torch.empty(batch_size, nz, device=device)
    mask_noise = torch.empty(batch_size, nz, device=device)

    lrate = 1e-4
    data_gen_optimizer = optim.Adam(
        data_gen.parameters(), lr=lrate, betas=(.5, .9))
    mask_gen_optimizer = optim.Adam(
        mask_gen.parameters(), lr=lrate, betas=(.5, .9))

    data_critic_optimizer = optim.Adam(
        data_critic.parameters(), lr=lrate, betas=(.5, .9))
    mask_critic_optimizer = optim.Adam(
        mask_critic.parameters(), lr=lrate, betas=(.5, .9))

    update_data_critic = CriticUpdater(
        data_critic, data_critic_optimizer, device, batch_size)
    update_mask_critic = CriticUpdater(
        mask_critic, mask_critic_optimizer, device, batch_size)

    '''
    MisGAN Training
    '''

    update_interval = 5
    critic_updates = 0

    data_losses = []
    mask_losses = []

    for epoch in range(1):

        print("epoch" + str(epoch))

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

            critic_updates += 1

            if critic_updates == n_critic:
                critic_updates = 0

                # Update generators' parameters
                for p in data_critic.parameters():
                    p.requires_grad_(False)
                for p in mask_critic.parameters():
                    p.requires_grad_(False)

                data_gen.zero_grad()
                mask_gen.zero_grad()

                data_noise.normal_()
                mask_noise.normal_()

                fake_data = data_gen(data_noise)
                fake_mask = mask_gen(mask_noise)
                masked_fake_data = mask_data(fake_data, fake_mask)

                data_loss = -data_critic(masked_fake_data).mean()
                data_loss.backward(retain_graph=True)
                data_gen_optimizer.step()

                mask_loss = -mask_critic(fake_mask).mean()
                (mask_loss + data_loss * alpha).backward()
                mask_gen_optimizer.step()

                for p in data_critic.parameters():
                    p.requires_grad_(True)
                for p in mask_critic.parameters():
                    p.requires_grad_(True)

        if (epoch + 1) % update_interval == 0:
            # Although it makes no difference setting eval() in this example,
            # you will need those if you are going to use modules such as
            # batch normalization or dropout in the generators.
            data_gen.eval()
            mask_gen.eval()

            data_losses.append(data_loss)
            mask_losses.append(mask_loss)

            print("MisGAN data loss = {0}".format(data_loss))
            print("MisGAN mask loss = {0}".format(mask_loss))

            data_gen.train()
            mask_gen.train()

    torch.save(data_gen.state_dict(), join('checkpoint', 'misgan_checkpoint', fname + '_data_gen.pth'))
    torch.save(mask_gen.state_dict(), join('checkpoint', 'misgan_checkpoint', fname + '_mask_gen.pth'))
    torch.save(data_critic.state_dict(), join('checkpoint', 'misgan_checkpoint', fname + '_data_critic.pth'))
    torch.save(mask_critic.state_dict(), join('checkpoint', 'misgan_checkpoint', fname + '_mask_critic.pth'))

    with open('result/misgan_result/training_misgan_loss.txt', 'wb') as f:
        f.write(str(zip(data_losses, mask_losses)))

    '''
    Imputer initialization
    '''

    imputer = Imputer(K, D).to(device)
    impu_critic = ConvCritic(K, D).to(device)
    impu_noise = torch.empty(batch_size, 1, K, D, device=device)

    imputer_lrate = 2e-4
    imputer_optimizer = optim.Adam(
        imputer.parameters(), lr=imputer_lrate, betas=(.5, .9))
    impu_critic_optimizer = optim.Adam(
        impu_critic.parameters(), lr=imputer_lrate, betas=(.5, .9))
    update_impu_critic = CriticUpdater(
        impu_critic, impu_critic_optimizer, device, batch_size)

    '''
    Imputer training
    '''

    beta = .1
    update_interval = 1
    critic_updates = 0
    data_losses = []
    mask_losses = []
    imputer_losses = []

    for epoch in range(1):

        print("epoch" + str(epoch))

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

            critic_updates += 1

            if critic_updates == n_critic:
                critic_updates = 0

                # Update generators' parameters
                for p in data_critic.parameters():
                    p.requires_grad_(False)
                for p in mask_critic.parameters():
                    p.requires_grad_(False)
                for p in impu_critic.parameters():
                    p.requires_grad_(False)

                data_noise.normal_()
                fake_data = data_gen(data_noise)

                mask_noise.normal_()
                fake_mask = mask_gen(mask_noise)
                masked_fake_data = mask_data(fake_data, fake_mask)

                impu_noise.uniform_()
                imputed_data = imputer(real_data, real_mask, impu_noise)

                data_loss = -data_critic(masked_fake_data).mean()
                mask_loss = -mask_critic(fake_mask).mean()
                impu_loss = -impu_critic(imputed_data).mean()

                mask_gen.zero_grad()
                (mask_loss + data_loss * alpha).backward(retain_graph=True)
                mask_gen_optimizer.step()

                data_gen.zero_grad()
                (data_loss + impu_loss * beta).backward(retain_graph=True)
                data_gen_optimizer.step()

                imputer.zero_grad()
                impu_loss.backward()
                imputer_optimizer.step()

                for p in data_critic.parameters():
                    p.requires_grad_(True)
                for p in mask_critic.parameters():
                    p.requires_grad_(True)
                for p in impu_critic.parameters():
                    p.requires_grad_(True)

        if (epoch + 1) % update_interval == 0:
            with torch.no_grad():
                imputer.eval()

                ###
                # inference mode
                # find out difference between gen_mask and real_mask
                ##
                # run inference mode

                data_losses.append(data_loss)
                mask_losses.append(mask_loss)
                imputer_losses.append(impu_loss)
                print("MisGAN data loss = {0}".format(data_loss))
                print("MisGAN mask loss = {0}".format(mask_loss))
                print("MisGAN imputer loss = {0}".format(imputer_loss))

                imputer.train()

    torch.save(imputer.state_dict(), join('checkpoint', 'misgan_checkpoint', fname + '_imputer.pth'))
    torch.save(impu_critic.state_dict(), join('checkpoint', 'misgan_checkpoint', fname + '_impute_critic.pth'))
    with open('result/misgan_result/training_imputer_loss.txt', 'wb') as f:
        f.write(str(zip(data_losses, mask_losses, imputer_losses)))
