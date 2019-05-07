from mlmi_cnn_relation_extraction import MLMI_CNN_Model


def main(input_file_path):
	model = MLMI_CNN_Model()
	model_data_path = model.read_dataset(input_file_path)
	model.data_preprocess(model_data_path)
	
	train_dir = model.train(model_data_path)
	print(train_dir)

	output_file_path = model.predict(model_data_path,trained_model = train_dir)
	model.evaluate(output_file_path)

	return output_file_path



if __name__ == '__main__':
	input_file_path = 'data/NYT/nyt_300.txt'

	output_file_path = main(input_file_path)
	print(output_file_path)

