

import numpy as np
from utils import _simulate_missing_data
from riddle.models.mlp import MLP
from utils import evaluate
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import mean_squared_error



def main(input_file_path):
    letters = pd.read_csv(input_file_path, dtype='float32', delimiter=',')

    x = np.array(letters.drop(['letter'], 1))

    y = np.array(letters['letter'])

    x = x.tolist()
    y = y.tolist()

    x = _simulate_missing_data(x, prop_missing=0.2)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=1)

    start = time.time()
    model = MLP(30, 26)
    model.train(x_train, y_train, x_val, y_val)
    y_probas = model.predict_proba(x_test)

    print(y_probas)
    y_pred = np.argmax(y_probas, axis=1)
    print(y_pred)

    runtime = time.time() - start
    out_dir = '/Users/ashiralam/riddle/_out_test/test_letter/'

    evaluate(y_test, y_probas, runtime, num_class=26,
             out_dir=out_dir)
    output_file_path = 'out.txt'
    with open(output_file_path, 'w') as f:
        print('Filename:', y_pred, file=f)


    print('Mean Squared Error: {:.4f}'.format(mean_squared_error(y_test, y_pred)))

    return output_file_path












