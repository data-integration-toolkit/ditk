import ReQuest
from ReQuest import ReQuest

obj=ReQuest()

input="data/source/NYT/test.json"

data=obj.read_dataset(input)

# Start the Stanford corenlp server for the python wrapper before executing this code.
# feature generation
obj.data_preprocess(data)

#Feature extraction, embedding learning on training data, and evaluation on test data.
obj.train()

#Evaluates relation extraction performance (precision, recall, F1): produce predictions.
obj.evaluate()