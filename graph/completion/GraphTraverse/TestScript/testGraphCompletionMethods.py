import unittest
import pandas as pd
#from graph.completion.graph_completion import graph_completion
#from dskg import *
#from demo1 import *
from demo2 import *
from graphTraverse import *



class TestGraphCompletionMethods():
    def __init__(self):
        self.obj=GraphTraverse()

    def test_read_dataset(self):
       try:
          self.obj.read_dataset([dataset_path,params_path,glove_path],"FreeBase")
          print("True")
       except:
           print("False")


    def test_predict(self):

        try:
            self.obj.train()
            print("True")

        except:
            print("False")

    def test_evaluate(self):
        try:
            self.obj.evaluate()
            print("True")
        except:
            print("False")

    def test_output_facts(self):
        try:
            self.obj.train()
            print("True")
        except:
            print("False")


if __name__ == '__main__':
    unittest.main()