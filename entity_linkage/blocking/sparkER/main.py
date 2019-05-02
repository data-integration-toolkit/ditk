from sparkER import SparkER
import csv

def main(input_file_path):
	output_file_path = "output.txt"
	with open(input_file_path) as f:
		dataset1 = f.readline().strip()
		dataset2 = f.readline().strip()
	sparker = SparkER()
	#dataset1 = "Dataset1.csv"
	#dataset2 = "Dataset2.csv"
	file_list = [dataset1, dataset2]
	dataframes = sparker.read_dataset(file_list)
	print(dataframes[0].head(5))
	sparker.train(dataframes)
	predicted_pair_list = sparker.predict(dataframe_list = dataframes)
	print(predicted_pair_list[0][10])
	groundtruth = "dataset-sample/articlesGround.csv"
	print(sparker.evaluate(groundtruth, dataframes))
	f = open(output_file_path, "w+b")
	c = csv.writer(f)
	c.writerows(predicted_pair_list[0])
	f.close()

	return output_file_path

if __name__ == '__main__':
    main("input.txt")
