from breds_relation_extraction import BREDSModel


def main(input_file_path):
	model = BREDSModel()
	
	test_data = model.read_dataset(input_file_path)
	model.train(input_file_path)

	output_file_path = model.predict(test_data)
	model.evaluate(output_file_path)

	return output_file_path



if __name__ == '__main__':
	input_file_path = 'data/semeval/semeval.txt'

	output_file_path = main(input_file_path)
	print(output_file_path)

