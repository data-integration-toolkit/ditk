import numpy as np
import csv
import os
from os.path import join
from tqdm import tqdm


def create_test_file(fname, outfile):
    with open(fname, 'r') as f:
        rows = csv.reader(f, delimiter=',', quotechar='|')
        data = [x for x in rows]
        header = np.asarray(data[0])
        data = np.asarray(data[1:], dtype='float')

    N, D = data.shape
    n = N * D
    ratio = 0.2
    k = int(n * ratio)
    arr = np.array([0] * k + [1] * (n-k))
    np.random.shuffle(arr)
    arr = np.reshape(arr, (N, D))

    data = np.multiply(arr, data)

    with open(outfile, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(header.tolist())
        csvwriter.writerows(data.tolist())

fpaths = []
outpaths = []
for file in tqdm(os.listdir("data")):
    fpath = join("data", file)
    fpaths.append(fpath)
    outpaths.append(join("data", "test_" + file))
for fpath, outpath in zip(fpaths, outpaths):
    create_test_file(fpath, outpath)
