import os
import sys
import unittest

import pandas as pd

PACKAGE_PARENT = '../../../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
print(os.getcwd())

from extraction.named_entity.cvt import main


class TestNERMethods(unittest.TestCase):
    file_root = os.getcwd()
    input_file = file_root + '/tests/sample_input.txt'
    output_file = ''

    def test_a_setup(self):

        print("files root is: ", self.__class__.file_root)
        self.__class__.output_file = main.main(self.__class__.input_file)

    def row_col_count(self, file_name):
        df = pd.read_csv(file_name, delimiter=r'\s+')
        return df.shape

    def test_outputformat(self):
        input_row_count = self.row_col_count(self.__class__.input_file)[0]
        input_col_count = self.row_col_count(self.__class__.input_file)[1]
        output_row_count = self.row_col_count(self.__class__.output_file)[0]
        output_col_count = self.row_col_count(self.__class__.output_file)[1]

        self.assertEqual(input_row_count, output_row_count)
        self.assertEqual(output_col_count, 3)
        self.assertEqual(input_col_count, 17)


if __name__ == '__main__':
    unittest.main()
