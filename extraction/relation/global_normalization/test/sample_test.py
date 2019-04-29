import unittest
import sys
import os
import pandas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relation_extraction_2 import RelationExtractionModel

class TestRelationExtraction(unittest.TestCase):

    def setUp(self):
        #instantiate the implemented class
        self.relation_extraction = RelationExtractionModel()
        self.outfile_file = ''

    def test_outputfile(self):
        #test if the third element in the output file is equal to 'was'
        pred_relation_file = 'results/CoNLL04_prediction.txt'
        df = pandas.read_csv(pred_relation_file, sep='\t')
        self.assertEqual(len(df.columns), 3)



if __name__ == '__main__':
    unittest.main()