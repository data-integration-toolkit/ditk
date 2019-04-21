from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')

from neuroner import neuroner


# create the model
nn = neuroner(parameters_filepath='./parameters.ini')
inputFiles = {'train':'/Users/lakshya/Desktop/CSCI-548/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs-master/conll/train.txt','dev':'/Users/lakshya/Desktop/CSCI-548/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs-master/conll/valid.txt','test':'/Users/lakshya/Desktop/CSCI-548/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs-master/conll/test.txt'}

datasetname = 'CONLL2003'
data = nn.read_dataset(inputFiles, datasetname)
#nn.train(data)
nn.load_model()
predictions = nn.predict_dataset(data)

P,R,F1 = nn.evaluate(predictions)  # calculate Precision, Recall, F1`

print('Precision: %s, Recall: %s, F1: %s'%(P,R,F1))
#metrics = nn.evaluate(predictions, ground_truth)
#nn.train(nn.modeldata)
#print(nn.predict("I love Chicago, IL"))
nn.close()
