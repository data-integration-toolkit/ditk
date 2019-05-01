from conec import conec

def main():
	context2vec = conec()
	context2vec.read_Dataset("text8", "data/text8")
	context2vec.train(iterations=1, saveInterm=False, modelType="cbow", embed_dim=200 )
	
	context2vec.predict_embedding("student")
	context2vec.predict_sent_embedding("This is a great class")
	context2vec.predict_similarity("wars", "war")
	context2vec.predict_sent_similarity("I am a girl", "I am a woman")
	
	context2vec.evaluate_analogy()

	context2ner = conec()
	context2ner.read_Dataset("conll2003", "data/conll2003/train.txt")
	context2ner.train(iterations=20, saveInterm=False, modelType="cbow", embed_dim=200)
	context2ner.evaluate_ner()

if __name__ == '__main__':
	main()