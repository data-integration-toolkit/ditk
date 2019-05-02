import numpy as np

from sklearn.metrics import accuracy_score
from riddle import emr
from riddle.models.mlp import MLP

from sklearn.model_selection import train_test_split

from sklearn.datasets import load_breast_cancer
from utils import evaluate
import time
from sklearn.metrics import mean_squared_error
from utils import _simulate_missing_data



def bc_evaluate():
    bc = load_breast_cancer()


    x = bc.data/4
    x = x.reshape([x.shape[0], -1])
    y = bc.target
    x = x.tolist()
    y = y.tolist()

    x = _simulate_missing_data(x, prop_missing=0.2)


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

    start = time.time()
    model = MLP(30,2)
    model.train(x_train, y_train, x_val, y_val)
    y_probas = model.predict_proba(x_test)
    y_pred = np.argmax(y_probas, axis=1)

    runtime = time.time() - start
    out_dir='/Users/ashiralam/riddle/_out_test/'


    evaluate(y_test, y_probas, runtime, num_class=2,
                 out_dir=out_dir)
    # print(y_test)
    # print(y_pred)
    print('Mean Squared Error: {:.4f}'.format(mean_squared_error(y_test, y_pred)))


if __name__ == '__main__':
    bc_evaluate()









