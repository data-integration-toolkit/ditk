import os,sys
from cnn_sts import STSTask,Embedder

def main(input_path):
	tsk = STSTask()
	tsk.load_resc('glove.840B.300d.txt')
	print("Loading Vectors - DONE!")
	print("Loading datasets....")

	path = os.getcwd()

	train_input = path+ "/data/sts2014/train.txt"

	val1 = path + "/data/sts2014/val.txt"

	test1 = path+ "/data/sts2014/test.txt"

	tsk.read_dataset(train_input, val1, test1)
	bestresult = 0.0
	bestwfname = None
	print("Done reading datasets!")
	print("Training")
	for i_run in range(tsk.c['num_runs']):
		print('RunID: %s' %i_run)
		print(i_run)
		tsk.train(i_run)
	tsk.save_model('cnn_model.h5')
	print("Model Saved!")
	print("Loading Model....")
	tsk.load_model('cnn_model.h5')
	print("reading input....")
	file = open(input_path,"r").readlines()
	sent1 = file[0].strip().replace("\"","")
	sent2 = file[1].strip().replace("\"","")
	score = str(round(tsk.predict(sent1,sent2),3))
	print("Similarity:" + str(round(score,3)))
	with open('./tests/sample_output.txt','w+') as output:
		output.write(str(score))

	output.close()

if __name__ == '__main__':
		# args = parser.parse_args()
		# pp.pprint(args)
	
	main("./tests/sample_input.txt")