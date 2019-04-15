import os
from os.path import isfile, join
import sys
from tqdm import tqdm


# change current folder to parent folder
parent = sys.path[0].rfind('/')
parent2 = sys.path[0].rfind('\\')

sys.path[0] = sys.path[0][:max(parent, parent2)]

from imputation import Imputation

import argparse
from misgan.modules import preprocess
from misgan.modules import train
from misgan.modules import test
from misgan.modules import evaluate
from misgan.modules import impute


class MisGAN(Imputation):
    def __init__(self, args):
        super(MisGAN, self).__init__()
        self.args = args

    def preprocess(self, *args, **kwargs):
        data = []
        fpaths = []
        for file in tqdm(os.listdir(join(os.pardir, "data"))):
            fpath = join(os.pardir, "data", file)
            if isfile(fpath):
                fpaths.append(file)
                data.append(super(MisGAN, self).preprocess(fpath, *args, **kwargs))
        preprocess.preprocess(self.args, data, fpaths)

    def train(self, *args, **kwargs):
        super(MisGAN, self).train(self.args.fname, *args, **kwargs)
        train.train(self.args.fname)

    def test(self, *args, **kwargs):
        super(MisGAN, self).test(self.args.model, self.args.fname, *args, **kwargs)
        test.test(self.args.model, self.args.fname)

    def impute(self, *args, **kwargs):
        _, self.impute_data = super(MisGAN, self).impute(self.args.model, join(os.pardir, self.args.fname), *args, **kwargs)
        impute.impute(self.args, self.args.model, self.impute_data)

    def evaluate(self, *args, **kwargs):
        _, self.eval_data = super(MisGAN, self).evaluate(self.args.model, join(os.pardir, self.args.fname), *args, **kwargs)
        self.model = 'wdbc_imputer.pth'
        evaluate.evaluate(self.args, self.args.model, self.eval_data)

    def load_model(self, *args, **kwargs):
        pass

    def save_model(self, *args, **kwargs):
        pass


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--fname',
                        help='name of the dataset file')
    parser.add_argument('--model',
                        help='name of the model')
    parser.add_argument('--ratio',
                        help='Ratio of the missing data')
    parser.add_argument('--split',
                        help='Ratio of training / testing split')
    parser.add_argument('--preprocess', action='store_true',
                        help='If set, run data pre-processing')
    parser.add_argument('--train', action='store_true',
                        help='If set, train model')
    parser.add_argument('--test', action='store_true',
                        help='If set, run testing mode')
    parser.add_argument('--evaluate', action='store_true',
                        help='If set, run inference mode')
    parser.add_argument('--impute', action='store_true',
                        help='If set, run inference mode without computing rmse')
    parser.add_argument('--misgan', action='store_true',
                        help='If set, set model to misgan')
    parser.add_argument('--imputer', action='store_true',
                        help='If set, set model to misgan imputer')
    parser.add_argument('--ims', action='store_true',
                        help='If set, introduce missing value and mask in parent class')
    args = parser.parse_args()

    misgan = MisGAN(args)

    print("Structure initialized")

    # creating path
    if not os.path.exists('data'):
        os.mkdir('data')

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')

    if not os.path.exists('result'):
        os.mkdir('result')

    if args.preprocess:
        misgan.preprocess(args)
    if args.train:
        misgan.train(args)
    if args.test:
        misgan.test(args)
    if args.evaluate:
        misgan.evaluate(args)
    if args.impute:
        misgan.impute(args)


if __name__ == '__main__':
    main()
