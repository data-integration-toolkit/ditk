from ditk_model import DITKModel
import argparse
import train
import test
import evaluate
import extract_vocab


class SQLQuery(DITKModel):
    def __init__(self, args):
        super(SQLQuery, self).__init__()
        self.args = args

    def train(self, *args, **kwargs):
        super(SQLQuery, self).train(*args, **kwargs)
        train.train(self.args)

    def extract_embedding(self, *args, **kwargs):
        super(SQLQuery, self).extract_embedding(*args, **kwargs)
        extract_vocab.extract_vocab()

    def test(self, *args, **kwargs):
        super(SQLQuery, self).test(*args, **kwargs)
        test.test(self.args)

    def evaluate(self, *args, **kwargs):
        super(SQLQuery, self).evaluate(*args, **kwargs)
        evaluate.evaluate(self.args)
        pass


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='If set, train model')
    parser.add_argument('--test', action='store_true',
                        help='If set, run testing mode')
    parser.add_argument('--evaluate', action='store_true',
                        help='If set, run inference mode')
    parser.add_argument('--extract_emb', action='store_true',
                        help='If set, extract glove embedding for training')
    parser.add_argument('--toy', action='store_true',
                        help='If set, use small data; used for fast debugging.')
    parser.add_argument('--suffix', type=str, default='',
                        help='The suffix at the end of saved model name.')
    parser.add_argument('--ca', action='store_true',
                        help='Use conditional attention.')
    parser.add_argument('--dataset', type=int, default=0,
                        help='0: original dataset, 1: re-split dataset')
    parser.add_argument('--maxepoch', action='store_true',
                        help='Max training epoch')
    parser.add_argument('--rl', action='store_true',
                        help='Use RL for Seq2SQL(requires pretrained model).')
    parser.add_argument('--baseline', action='store_true',
                        help='If set, then train Seq2SQL model; default is SQLNet model.')
    parser.add_argument('--train_emb', action='store_true',
                        help='Train word embedding for SQLNet(requires pretrained model).')
    args = parser.parse_args()

    sqlQuery = SQLQuery(args)

    if args.extract_emb:
        sqlQuery.extract_embedding(args, None)
    if args.train:
        sqlQuery.train(args, None)
    if args.test:
        sqlQuery.test(args, None)
    if args.evaluate:
        sqlQuery.evaluate(args, None)


if __name__ == '__main__':
    main()
