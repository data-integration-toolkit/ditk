import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
from os.path import join
import numpy as np
from modules.misgan_modules import preprocess
import pickle
from tqdm import tqdm
from time import gmtime, strftime

from misgan_modules.misgan_nn import *

BASE_PATH = 'checkpoint/misgan_checkpoint'
OUT_BASE = 'result/misgan_result'


def getrmse(table, original_data, mask):
    mask = mask == 0
    imputed_val = table[mask]
    original_data = original_data[mask]

    return np.sqrt(np.mean((imputed_val - original_data)**2))


def construct_full_table(result, original_data, K, D, row_size, col_size):
    fragments = {}

    # 1 is used as default stride
    # create the imputed table
    table = np.zeros(((row_size - 1) * 1 + K, (col_size - 1) * 1 + D))

    # set the new table to have same value of the original table
    for i in range(n):
        for j in range(d):
            table[i][j] = original_data[i][j]

    # apply imputation values
    for data in tqdm(result):
        real_mask = data[1].cpu().numpy()
        impute_data = data[2].cpu().detach().numpy()
        indices = data[3].cpu().detach().numpy()

        rows = [int(x / col_size) for x in indices]
        cols = [int(x % y) for x, y in zip(indices, rows)]

        # map phase, not using spark
        for k in range(len(real_mask)):
            for r, c in zip(rows, cols):
                for x in range(K):
                    for y in range(D):
                        if real_mask[k][0][x][y] == 0:
                            key = str(x + r) + ',' + str(y + c)
                            if key not in fragments:
                                fragments[key] = [impute_data[k][0][x][y]]
                            else:
                                fragments[key].append(impute_data[k][0][x][y])
    # reduce
    for key, val in fragments.items():
        fragments[key] = np.sum(val) / float(len(val))

    # create imputed table
    for key, val in fragments.items():
        row, col = np.array(key.split(','), dtype=int)
        table[row][col] = val

    return table


def impute(args, model, impute_data):

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    data_loader = preprocess.preprocess(args, [(impute_data, '', None)], ['impute_data'], save=False)

    '''
    Imputer initialization
    '''
    K, D = 16, 16
    imputer = Imputer(K, D).to(device)
    imputer.load_state_dict(torch.load(join(BASE_PATH, model)))
    imputer.eval()

    result = []
    for real_data, real_mask, index in data_loader[0]:
        # find out if input of any size works too
        real_data = real_data.to(device)
        real_mask = real_mask.to(device).float()

        impute_data = imputer.inference(real_data, real_mask)
        result.append([real_data, real_mask, impute_data, index])

    table = construct_full_table(result, data_loader[0].dataset.original_data, K, D, row_size, col_size)

    # create path if it does not exist
    if not os.path.exists('result'):
        os.mkdir('result')
    if not os.path.exists('result/misgan_result'):
        os.mkdir('result/misgan_result')

    # write to file
    outfile = join(OUT_BASE, strftime("%d_%b_%Y %H:%M:%S", gmtime()) + 'impute_data.csv')
    with open(outfile, 'rb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in table:
            csvwriter.writerow(row)

    print("CSV Saved")

