import os
import sys
import unittest
import pandas as pd

if os.name == 'nt':
    module_path = os.path.abspath(os.path.join('..\..\..\..'))
else:
    module_path = os.path.abspath(os.path.join('../../../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from entity_linkage.typing.oneShotRelationalLearning.one_shot_relational_learning import OneShotRelationalLearning
from entity_linkage.typing.oneShotRelationalLearning.main import main

class TestEntityTypingMethods(unittest.TestCase):

    def setUp(self):
        self.et = OneShotRelationalLearning() #Your implementation of Entity Typing
        self.input_file = './sample_input.txt'
        self.output_file = main(self.input_file)

    def test_read_dataset(self) :
        return_dataset = self.et.read_dataset([self.input_file])
        self.assertTrue(os.path.isdir(return_dataset))
        self.assertTrue(os.path.exists(return_dataset + "/path_graph")) # check graph was generated

    def test_evaluate(self) :
        return_dataset = self.et.read_dataset([self.input_file])
        eval_result = self.et.evaluate(return_dataset)
        self.assertTrue(os.path.exists(eval_result))
        content = None
        with open (eval_result) as evaluate_result_file :
            content = evaluate_result_file.read()

        self.assertTrue("MRR" in content)
        self.assertTrue("HITS10" in content)
        self.assertTrue("HITS5" in content)

if __name__ == '__main__':
	unittest.main()
