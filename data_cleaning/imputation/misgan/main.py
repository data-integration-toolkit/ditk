from Imputation_3 import Imputation
import argparse
import os
from os.path import isfile, join
from modules.misgan_modules import preprocess
from modules.misgan_modules import train
from modules.misgan_modules import test
from modules.misgan_modules import evaluate
from modules.misgan_modules import impute


class MisGAN(Imputation):
    def __init__(self, args):
        super(MisGAN, self).__init__()
        self.args = args

    def preprocess(self, *args, **kwargs):
        data = []
        fpaths = []
        for file in os.listdir("data"):
            fpath = join("data", file)
            if isfile(fpath):
                fpaths.append(file)
                data.append(super(MisGAN, self).preprocess(fpath, *args, **kwargs)[0])
        preprocess.preprocess(self.args, data, fpaths)

    def train(self, *args, **kwargs):
        super(MisGAN, self).train(self.args.fname, *args, **kwargs)
        train.train(self.args.fname)

    def test(self, *args, **kwargs):
        self.model, self.test_data = super(MisGAN, self).test(self.args.model, self.args.fname, *args, **kwargs)
        if self.args.misgan:
            test.test(self.args, self.model, self.test_data)
        if self.args.imputer:
            test.test(self.args, self.model, self.test_data)

    def impute(self, *args, **kwargs):
        self.model, self.impute_data = super(MisGAN, self).impute(self.args.model, self.args.fname, *args, **kwargs)
        impute.impute(self.args, self.model, self.impute_data)

    def evaluate(self, *args, **kwargs):
        self.model, self.eval_data = super(MisGAN, self).evaluate(self.args.model, self.args.fname, *args, **kwargs)
        self.model = 'wdbc_imputer.pth'
        evaluate.evaluate(self.args, self.model, self.eval_data)


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
