
from ner_blstm_cnn import ner_blstm_cnn

inputFiles = {'train':'/Users/lakshya/Desktop/CSCI-548/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs-master/conll/train.txt','dev':'/Users/lakshya/Desktop/CSCI-548/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs-master/conll/valid.txt','test':'/Users/lakshya/Desktop/CSCI-548/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs-master/conll/test.txt'}

# instatiate the class
myModel = ner_blstm_cnn(1)

# read in a dataset for training
data = myModel.read_dataset(inputFiles)

# trains the model and stores model state in object properties or similar
myModel.train(data)

# generate predictions output format will be same for everyone
predictions = myModel.predict(data['test'])


P,R,F1 = myModel.evaluate(predictions)  # calculate Precision, Recall, F1

print('Precision: %s, Recall: %s, F1: %s'%(P,R,F1))