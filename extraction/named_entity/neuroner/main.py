'''
To run:
CUDA_VISIBLE_DEVICES="" python3.5 main.py &
CUDA_VISIBLE_DEVICES=1 python3.5 main.py &
CUDA_VISIBLE_DEVICES=2 python3.5 main.py &
CUDA_VISIBLE_DEVICES=3 python3.5 main.py &
'''
from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')

from neuroner import neuroner


# create the model
nn = neuroner(parameters_filepath='../parameters.ini')
filedict = {"train":"./data/conll2003/en/train.txt", "valid":"./data/conll2003/en/valid.txt", "test":"./data/conll2003/en/test.txt"}

datasetname = 'CONLL2003'
data = nn.read_dataset(filedict, datasetname)
#nn.train(data)
ground_truth = nn.convert_ground_truth(data)
predictions = nn.predict_dataset(data)

P,R,F1 = nn.evaluate(predictions, ground_truth)  # calculate Precision, Recall, F1`

print('Precision: %s, Recall: %s, F1: %s'%(P,R,F1))
#metrics = nn.evaluate(predictions, ground_truth)
#nn.train(nn.modeldata)
#print(nn.predict("I love Chicago, IL"))
nn.close()
