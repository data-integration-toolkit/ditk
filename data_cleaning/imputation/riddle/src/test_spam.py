import numpy as np

from sklearn.metrics import accuracy_score
from riddle import emr
from riddle.models.mlp import MLP
from sklearn.model_selection import train_test_split
import pandas as pd
import time
from utils import evaluate
import csv
from numpy.random import RandomState
from sklearn.metrics import mean_squared_error

def spam_evaluate():
    data = []
    f = open('spambase.txt')
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        data.append(row)
    f.close()

    x = np.array([x[:-1] for x in data]).astype(np.float)
    y = np.array([x[-1] for x in data]).astype(np.float)
    del data

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

    start = time.time()
    model = MLP(57, 2)
    model.train(x_train, y_train, x_val, y_val)
    y_probas = model.predict_proba(x_test)


    print(y_probas.shape)
    y_pred = np.argmax(y_probas, axis=1)
    print(y_pred.shape)


    runtime = time.time() - start
    out_dir='/Users/ashiralam/riddle/_out_test/'


    evaluate(y_test, y_probas, runtime, num_class=2,
                 out_dir=out_dir)
    # print(y_test)
    # print(y_pred)
    print('Mean Squared Error: {:.4f}'.format(mean_squared_error(y_test, y_pred)))

if __name__ == '__main__':
    spam_evaluate()

