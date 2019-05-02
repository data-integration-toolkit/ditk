from blast import Blast
import csv

def main(input_file_path):
	output_file_path = "output.txt"
	with open(input_file_path) as f:
		dataset1 = f.readline().strip()
		dataset2 = f.readline().strip()
	blast = Blast()
	#dataset1 = "Dataset1.csv"
	#dataset2 = "Dataset2.csv"
	file_list = [dataset1, dataset2]
	dataframes = blast.read_dataset(file_list)
	print(dataframes[0].head(5))
	blast.train(dataframes)
	predicted_pair_list = blast.predict(dataframe_list = dataframes)
	print(predicted_pair_list[0][10])
	groundtruth = "dataset-sample/articlesGround.csv"
	print(blast.evaluate(groundtruth, dataframes))
	f = open(output_file_path, "w+b")
	c = csv.writer(f)
	c.writerows(predicted_pair_list[0])
	f.close()

	return output_file_path

if __name__ == '__main__':
    main("input.txt")
