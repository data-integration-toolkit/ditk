from conec import conec

def main(inputFile, outputFile):
	context2vec = conec()
	#For running the tests
	print("Reading sample inputs from tests/sample_input.txt")
	file = open(inputFile, "r")
	out = open(outputFile, "w+")
	words = file.readlines()
	for word in words:
		word = word.strip()
		text = "{} \n {}\n".format(word, context2vec.predict_embedding(word))
		# print(text)
		out.write(text)
	print("Written output to test/sample_output.txt")

	print("Training on text8 ..........")
	context2vec.read_Dataset("text8", "data/text8")
	context2vec.train(iterations=1, saveInterm=False, modelType="cbow", embed_dim=200 )
	
	context2vec.predict_embedding("student")
	context2vec.predict_sent_embedding("This is a great class")
	context2vec.predict_similarity("wars", "war")
	context2vec.predict_sent_similarity("I am a girl", "I am a woman")
	
	print("Evaluating model on Google Analogy Dataset ...")
	context2vec.evaluate_analogy()

	print("Training and Evaluating on CoNLL 2003 ...")
	context2ner = conec()
	context2ner.read_Dataset("conll2003", "data/conll2003/train.txt")
	context2ner.train(iterations=20, saveInterm=False, modelType="cbow", embed_dim=200)
	context2ner.evaluate_ner()

if __name__ == '__main__':
	file_in = "tests/sample_input.txt"
	file_out = "tests/sample_output.txt"

	main(file_in, file_out)