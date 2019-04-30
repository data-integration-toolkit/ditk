from model.main import DeepElmoEmbedNer

deen = DeepElmoEmbedNer()

file_dict = dict()
file_dict['train'] = '../data/sample/ner_test_input.txt'
file_dict['test'] = '../data/sample/ner_test_input.txt'
file_dict['dev'] = '../data/sample/ner_test_input.txt'

data = deen.read_dataset(file_dict, "CoNLL2003")
model, sess, saver = deen.train(data, None, maxEpoch=1)
deen.predict('../data/sample/ner_test_input.txt', writeInputToFile=False, model=model, sess=sess, saver=saver, trainedData=data['train'])
deen.evaluate(None, None, None)

