import unittest
import pandas as pd
import sys
sys.path.append('..')
from paper1_modified import End_to_end as ner
import configs      
## this configs file serves specifically for the unittest part, so make sure your current directory is under the 'test' folder


## Before the unittest, please make sure you have created a 'DATA' folder under the current directory
## and put the glove.6b.100d embedding file, training and validation file under the 'DATA' folder
## then put the file to be tested under the current directory, its name should be 'ner_test_input.txt'
## the output file's name will be 'ner_test_output.txt'

## If you don't want to create the new folder or name the files in the way above, just change those
## parameters' value in configs.py which is under the current 'test' folder.

## Also, make sure you have at least two space lines at the end of your test file, otherwise the row
## numbers could not be the same!! 

class TestNERMethods(unittest.TestCase):

    def setUp(self):
        filenames = [configs.TRAINING_FILE, configs.VALIDATION_FILE]
        self.input_file = configs.TEST_FILE
        data1 = ner.read_dataset(filenames)
        ner.train(data1)
        data2 = ner.predict()
        self.output_file =  configs.OUTPUT_FILE

    def row_col_count(self,file_name):
        df = pd.read_csv(file_name,delimiter=' ')
        return df.shape

    def test_outputformat(self):
        print(self.input_file)
        print(self.output_file)
        input_row_count = self.row_col_count(self.input_file)[0]
        input_col_count = self.row_col_count(self.input_file)[1]
        output_row_count = self.row_col_count(self.output_file)[0]
        output_col_count = self.row_col_count(self.output_file)[1]
        print('input: ' + str(input_row_count)+ ', '+ str(input_col_count+1))
        print('output: ' + str(output_row_count)+ ', '+ str(output_col_count ))

        self.assertEqual(input_row_count, output_row_count)
        self.assertEqual(3, output_col_count)

if __name__ == '__main__':
    unittest.main()

