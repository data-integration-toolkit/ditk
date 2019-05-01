from Wikipedia2vec import Wikipedia2vec

def main():
	"""
	# Sample workflow:

	inputFiles = ['thisDir/file1.txt','thatDir/file2.txt','./file1.txt']

	myModel = Wikipedia2vec()  # instantiate the class

	data_X, Wikipedia_dump_file = myModel.read_dataset(inputFiles)  # read in dataset
	Wikipedia_dump_file = wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2  # download Wikipedia dump

	dump_db = myModel.build_dump(Wikipedia_dump_file)  # builds database that contains Wikipedia pages each of which consists of texts and anchor links in it
	output_dic = myModel.build_dictionary(inputFiles)  # builds a dictionary of entities from the input data.
	output_lg = myModel.build_link_graph(dump_db, output_dic)  # generates a sparse matrix representing the link structure between Wikipedia entities
	output_mention_db = myModel.build_mention_db(dump_db, output_dic)  # builds a database that contains the mappings of entity names (mentions) and their possible referent entities
	model_file = myModel.train_embedding(dump_db, output_dic)  # trains embeddings
	output_file = myModel.save_text(model_file)  # saves model_file results from train_embedding function in text format

	cosine_similarity = myModel.evaluate(output_file)  # calculate Cosine Similarity

	print('Cosine Similarity: %s'%(cosine_similarity))

	"""
	pre_trained = False
	obj = Wikipedia2vec()
	if pre_trained:
		obj.read_dataset()
		filename, embeddings = obj.load_model()
		ans = obj.evaluate(filename)
		print ans
	else:
		#dump_file = 'enwiki-latest-pages-articles.xml.bz2'
		#obj.read_dataset()
		#obj.build_dump(dump_file, 'output.db')
		#obj.build_dictionary('output.db', 'output_dic')
		#obj.build_link_graph('output.db', 'output_dic', 'output_lg')
		#obj.build_mention_db('output.db', 'output_dic', 'output_md')
		#obj.train_embedding('output.db', 'output_dic', 'final_output')
		#obj.save_text('final_output', 'final_output_text')
		ans = obj.evaluate('final_output_text')
		print ans
	

if __name__ == '__main__':
    main()