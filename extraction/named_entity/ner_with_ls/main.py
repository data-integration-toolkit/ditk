import os, sys

if os.name == 'nt':
	module_path = os.path.abspath(os.path.join('..\..\..'))
else:
	module_path = os.path.abspath(os.path.join('../../..'))

if module_path not in sys.path:
	sys.path.append(module_path)

from temp_subclass import NER_with_LS
import os, joblib, re, pyhocon, warnings, copy, sys

def main(input_file_path):

	# mandatory Inputs: (1)dataset_name, (2)ratio
	if "ontonotes" in input_file_path:
		dataset_name = "ontonotes"
	else:
		dataset_name = "conll"
	# ratio = (0.70, 0.15, 0.15)
	ratio = (0.0, 0.0, 1.0)

	options = {}
	options["ratio"] = ratio
	options["dataset_name"] = dataset_name

	# 1. Create my model
	print("==Created myModel...")
	myModel = NER_with_LS(dataset_name)
	file_dict = dict() # dict for data_path
	file_dict["train"] = myModel.config.raw_path+"/"+dataset_name+".train.txt"
	file_dict["dev"] = myModel.config.raw_path+"/"+dataset_name+".dev.txt"
	file_dict["test"] = myModel.config.raw_path+"/"+dataset_name+".test.txt"

	# Split data to 3 parts
	myModel.split_data_txt(input_file_path, file_dict, options)
	
	# 2. Read dataset for training
	print("==Reading dataset")
	data = myModel.read_dataset(file_dict)

	# 3. Train the model 
	print("==Training myModel w/ training dataset...")
	myModel.train(data)

	# 4-1. saved model
	print("==Saved model...")
	myModel.save_model()

	# 4-2. restore model
	print("==Load model...")
	myModel.load_model()

	# 5. Predict
	print("==Predict Test data...")
	pred_labels = myModel.predict(data["test"])
	print("len of pred_labels: ", len(pred_labels))

	# 6. Get truth labels
	print("==Getting ground truth data...")
	ground_truth_labels = myModel.convert_ground_truth(data["test"])
	print("len of ground_truth_labels: ", len(ground_truth_labels))

	# 7. Evaluate test input data
	print("==Evaluate test data...")
	scores = myModel.evaluate(pred_labels, ground_truth_labels)
	print("==precision, recall, f1")
	print(scores)

	output_file_path = myModel.config["output_path"] + ".test.output"
	return output_file_path


if __name__ == "__main__":

	input_file_path = "./test/conll2003_sample.txt" # conll2003 data sample
	# input_file_path = "./test/ontonotes_sample.txt" # ontonotes(conll2012) data sample
	output_file_path = main(input_file_path)
	print(output_file_path)




