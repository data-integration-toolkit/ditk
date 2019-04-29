from ner_gcn import GcnNer

def main():
	model = GcnNer()
	fileDict = {'train': './data/train.conll', 'dev': './data/dev.conll', 'test': './data/test.conll'}
	model.train_data, model.test_data = model.read_dataset(file_dict = fileDict, dataset_name = "")
	print("Starting training...")
	model.train(data = model.train_data, epochs = 2)
	print("Training completed")
	ground_truth_tuples = model.convert_ground_truth(model.predict_data, fileDict['dev'])
	ner = model.load_model("./data/ner-gcn-21.tf")
	entity_tuples = model.predict('./data/input.txt', pretrained_model = ner)
	output = open("gcn_ner_predictions.txt", "w")
	output.write("WORD TRUE_LABEL PRED_LABEL"+"\n\n")
	for each in entity_tuples:
		str = ""
		str = each[0]+" "+each[1]+" "+each[2]
		output.write(str+"\n")
		if (each[0] == "/." or each[0] == "/-") and entity_tuples.index(each) != len(entity_tuples)-1:
			output.write("\n")
	output.close()
	file = open("gcn_ner_predictions.txt", "r")
	lines = file.read()
	lines = lines.rstrip("\n")
	file.close()
	output = open("gcn_ner_predictions.txt", "w")
	output.write(lines)
	output.close()
	print("Output of predict stored at ./gcn_ner_predictions.txt")
	(precision, recall, f1) = model.evaluate(predictions = entity_tuples, groundTruths = ground_truth_tuples, pretrained_model = ner)
	print("*************Evaluation Metrics*************")
	print("Precision: ", precision*100)
	print("Recall: ",recall*100)
	print("F1 score: ", f1*100)

if __name__ == '__main__':
	main()
