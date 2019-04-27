import sys, os

# set path to ditk root
ditk_path = os.path.abspath(os.getcwd())
if ditk_path not in sys.path:
    sys.path.append(ditk_path)

from graph.completion.longae.longae import longae
from graph.completion.longae.hparams import hparams as hp
from sklearn.metrics import roc_auc_score as auc_score
from sklearn.metrics import average_precision_score as ap_score

# evaluation metric dictionary
metrics = {
    "auc_score": auc_score,
    "ap_score": ap_score
}

# construct main program
graph_completion = longae()

data_dir = os.path.join(ditk_path, 'graph/completion/longae/data/')
files_citeseer = list(map(lambda x: os.path.join(data_dir, x), ['citeseer_x.txt', 'citeseer_y.txt', 'citeseer_graph.txt']))
files_cora = list(map(lambda x: os.path.join(data_dir, x), ['cora_x.txt', 'cora_y.txt', 'cora_graph.txt']))

# set hparam file paths
hp.checkpoint_dir = os.path.join(ditk_path, 'graph/completion/longae/checkpoint/')
hp.index_file = os.path.join(ditk_path, 'graph/completion/longae/data/ind.cora.test.index')

train_data, validation_data, test_data = graph_completion.read_dataset(files_cora)
# graph_completion.load_model("./longae/checkpoint/checkpoint_100.h5")

graph_completion.train(train_data, validation_data)
print("Running test set")
prediction_data = graph_completion.predict(test_data)
graph_completion.evaluate(test_data, metrics, prediction_data)