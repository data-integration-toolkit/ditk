import subprocess

from model.main import BiLSTMCRFSeqTag
from util.ditk_convertor_util import convert_data_to_ditk

blcst = BiLSTMCRFSeqTag()

file_dict = dict()
file_dict['train'] = '../data/example/tester.txt'
file_dict['test'] = '../data/example/tester.txt'
file_dict['dev'] = '../data/example/tester.txt'

data = blcst.read_dataset(file_dict, "CoNLL2003")
blcst.train(data)
blcst.predict('../data/example/tester.txt', writeInputToFile=False)
blcst.evaluate(None, None, None)


