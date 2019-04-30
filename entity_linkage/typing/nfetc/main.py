import os
import sys

if os.name == 'nt':
	module_path = os.path.abspath(os.path.join('..\..\..'))
else:
	module_path = os.path.abspath(os.path.join('../../..'))

if module_path not in sys.path:
	sys.path.append(module_path)
	

from entity_tpying_subclass import NFETC

def main(input_file_path):

	print("> Creating my model...")
	myModel = NFETC()

	file_path = input_file_path
	# folder_path = "/".join(file_path.split("/")[:-1])
	data_name = "others"
	model_name = "best_nfetc_wiki"
	# ratio = (0.7, 0.15, 0.15)
	ratio = (0.0, 0.0, 1.0)
	epoch_num = 5

	# Mandatory options for my Model
	options = {}
	options["data_name"] = data_name
	options["ratio"] = ratio
	options["model_name"] = model_name
	options["epoch_num"] = epoch_num
	# model_names = {
	#	 "nfetc": param_space_nfetc,
	#	 "best_nfetc_wiki": param_space_best_nfetc_wiki,
	#	 "best_nfetc_wiki_hier": param_space_best_nfetc_wiki_hier,
	#	 "best_nfetc_ontonotes": param_space_best_nfetc_ontonotes,
	#	 "best_nfetc_ontonotes_hier": param_space_best_nfetc_ontonotes_hier,
	# }

	print("> Reading dataset ...")
	extension = file_path.split(".")[-1]
	if extension == "txt":
		myModel.split_data_txt(file_path, data_name, ratio)
	elif extension == "tsv":
		myModel.split_data_tsv(file_path, data_name, ratio)

	myModel.preprocess_helper(data_name, extension, input_file_path)
	train_data, test_data = myModel.read_dataset(file_path, options)

	# print("> Training ...") 
	# myModel.train(train_data, options) # saved trained model

	# print("> Save model ... ")
	# myModel.save_model()

	print(">Load Model ... ")
	myModel.load_model()
	
	print("> Predicting ...")
	predict_data = myModel.predict(test_data, None, options) 

	print("> Evaluating ...")
	acc, macro, micro = myModel.evaluate(test_data, predict_data, options)

	

	output_file_path = "./output/" + model_name + ".tsv"
	return output_file_path


if __name__ == "__main__":

	input_file_path = "./test/clean_data.tsv" # sample filtered data
	output_file_path = main(input_file_path)

	print(output_file_path)
