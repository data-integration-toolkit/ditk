import os
from os.path import isfile, join
import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class BlockMaskedMNIST(Dataset):
    def __init__(self, data, original_data, K, D, sx, sy, block_len=12, random_seed=0, ratio=None):
        self.K = K
        self.D = D
        self.row_size = sx
        self.col_size = sy
        self.block_len = block_len
        self.rnd = np.random.RandomState(random_seed)
        self.original_data = original_data
        data = torch.from_numpy(data)
        self.data_size = len(data)
        if ratio:
            self.block_len = np.floor(np.sqrt(ratio * self.data_size))

        self.generate_incomplete_data(data)

    def __getitem__(self, index):
        # return index so we can retrieve the mask location from self.mask_loc
        return self.image[index], self.mask[index], index

    def __len__(self):
        return self.data_size

    def generate_incomplete_data(self, data):
        n_masks = self.data_size
        self.image = [None] * n_masks
        self.mask = [None] * n_masks
        self.mask_loc = [None] * n_masks
        self.original_datamask = np.ones((self.original_data.shape))
        self.original_datamask_mapping = {}
        for i in range(n_masks):
            d0 = self.rnd.randint(0, self.K - self.block_len + 1)
            d1 = self.rnd.randint(0, self.D - self.block_len + 1)
            r, c = int(i / self.col_size), int(i % self.col_size)
            mask = np.zeros((self.K, self.D), dtype=np.uint8)
            mask[d0:(d0 + self.block_len), d1:(d1 + self.block_len)] = 1
            self.set_original_datamask(mask, r, c)
            mask = torch.from_numpy(mask)
            self.mask[i] = mask.unsqueeze(0)   # add an axis for channel
            self.mask_loc[i] = d0, d1, self.block_len, self.block_len
            # Mask out missing pixels by zero
            self.image[i] = data[i] * mask.float()

        # process original datamask mapping and find mask
        for key, val in self.original_datamask_mapping.items():
            row, col = np.array(key.split(','), dtype=int)
            self.original_datamask[row][col] = 0

    def set_original_datamask(self, mask, r, c):
        for x in range(self.K):
            for y in range(self.D):
                if mask[x][y] == 0:
                    key = str(x + r) + ',' + str(y + c)
                    self.original_datamask_mapping[key] = 0

class ArbitaryMasked(Dataset):
    def __init__(self, data, original_data, mask, K, D, sx, sy, random_seed=0):
        self.K = K
        self.D = D
        self.row_size = sx
        self.col_size = sy
        self.rnd = np.random.RandomState(random_seed)
        self.original_data = original_data
        data = torch.from_numpy(data)
        self.data_size = len(data)
        self.generate_dataholder(data, mask)

    def __getitem__(self, index):
        # return index so we can retrieve the mask location from self.mask_loc
        return self.image[index], self.mask[index], index

    def __len__(self):
        return self.data_size

    def generate_dataholder(self, data, mask):
        n_masks = self.data_size
        self.image = [None] * n_masks
        self.mask = [None] * n_masks
        self.mask_loc = [None] * n_masks
        mask_loc = mask == 0
        self.original_datamask = np.ones((self.original_data.shape))
        self.original_datamask[mask_loc] = 0
        for i in range(n_masks):
            r, c = int(i / self.col_size), int(i % self.col_size)
            mask = np.zeros((self.K, self.D), dtype=np.uint8)
            data_loc = mask[r: r+self.K, c: c+self.D] != 0
            mask[data_loc] = 1
            mask = torch.from_numpy(mask)
            self.mask[i] = mask.unsqueeze(0)  # add an axis for channel
            # Mask out missing pixels by zero
            self.image[i] = data[i] * mask.float()

# preprocesses data with manual introducing missing value in k x k blocks
def preprocess(args, full_data, fnames, save=True):
    if not os.path.exists('checkpoint/misgan_checkpoint'):
        os.mkdir('checkpoint/misgan_checkpoint')

    data_loaders = []

    for data_var, fname in zip(full_data, fnames):
        for dt in data_var:
            data = dt[0]
            prefix = dt[1]
            mask = dt[2]

            N, N0 = data.shape
            K, D = 16, 16
            padx, pady = 1, 1

            sx = int(((N - K) / padx) + 1)
            sy = int(((N0 - D) / pady) + 1)
            holder = [[[0]] * 1] * sx * sy

            # initialize holder to be 16x16 blocks with original data filled in
            for i in range(sx):
                for j in range(sy):
                    holder[i * sy + j][0] = data[i:i + K][:, j:j + D]

            if args.ims:
                data = ArbitaryMasked(np.asarray(holder, dtype='float32'), data, mask, K, D, sx, sy)
            else:
                data = BlockMaskedMNIST(np.asarray(holder, dtype='float32'), data, K, D, sx, sy, block_len=8, ratio=args.ratio)
            batch_size = 10
            data_loader = DataLoader(data, batch_size=batch_size, shuffle=True,
                                     drop_last=True)
            data_loaders.append(data_loader)

            if not os.path.exists("data/data_misgan"):
                os.mkdir("data/data_misgan")

            if save:
                dt_loader_path = join("data/data_misgan/", "{0}_{1}.data_loader".format(fname, prefix))
                with open(dt_loader_path, 'wb') as pickle_file:
                    pickle.dump(data_loader, pickle_file)

    return data_loaders