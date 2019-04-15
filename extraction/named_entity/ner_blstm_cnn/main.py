
from ner_blstm_cnn import ner_blstm_cnn

inputFiles = ['thisDir/file1.txt','thatDir/file2.txt','./file1.txt']

# instatiate the class
myModel = ner_blstm_cnn()

# read in a dataset for training
data = myModel.read_dataset(inputFiles)

# trains the model and stores model state in object properties or similar
myModel.train(data['train'])

# generate predictions output format will be same for everyone
predictions = myModel.predict(data['test'])

test_labels = myModel.convert_ground_truth(data['test'])

P,R,F1 = myModel.evaluate(predictions, test_labels)  # calculate Precision, Recall, F1

print('Precision: %s, Recall: %s, F1: %s'%(P,R,F1))

'''
filedict = {"train":"./chemdner/train.txt", "valid":"./chemdner/valid.txt", "test":"./chemdner/test.txt"}
dataset_name = 'ONTONOTES'

test_class = NER_LSTM_CNN(50)

data = test_class.read_dataset(filedict, dataset_name)



test_class.train(data)
test_class.load_models("./models/")


predictions = test_class.predict(data['test'])  # generate predictions! output format will be same for everyone`

test_labels = test_class.convert_ground_truth(data['test'])  #need ground truth labels need to be in same format as predictions!

P,R,F1 = test_class.evaluate(predictions, test_labels)  # calculate Precision, Recall, F1`

print('Precision: %s, Recall: %s, F1: %s'%(P,R,F1))
'''
#ground_truth = test_class.convert_ground_truth(data[2])


#test_class.load_models("./models/")
#predictions = test_class.predict_dataset(data[2])


#test_class.evaluate(predictions, ground_truth)

#print(test_class.predict("Steve went to Paris"))

#from ner import Parser

#p = Parser()

#p.load_models("models/")

#print(p.predict("Steve went to Paris"))