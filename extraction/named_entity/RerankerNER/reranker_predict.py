import os
def main(input_file):
	os.system('./CRF++-0.58/crf_test -m ./CRF++-0.58/model_file input_file >> output.txt')
	return 'output.txt'