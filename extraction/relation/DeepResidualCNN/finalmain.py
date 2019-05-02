from ResidualCNN9 import DeepResidualLearning

# train_file="relation_extraction_test_input.txt"
# traindata=DeepResidualLearning.DeepResidualLearningRE.read_dataset(train_file)
# print(type(traindata))
# # DeepResidualLearning.DeepResidualLearningRE.train(traindata)
test_file="relation_extraction_test_input.txt"
testdata=DeepResidualLearning.DeepResidualLearningRE.read_dataset(test_file)
DeepResidualLearning.DeepResidualLearningRE.predict(testdata)
