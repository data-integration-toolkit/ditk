import RelationExtractionCNN
import configure


train_file="relation_extraction_test_input.txt"
traindata=RelationExtractionCNN.DeepResidualLearningRE.read_dataset(train_file)
print(type(traindata))
# RelationExtractionCNN.DeepResidualLearningRE.train(traindata)
RelationExtractionCNN.DeepResidualLearningRE.predict(traindata)
