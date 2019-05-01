import conversionutil1
import model.train as train
import model.eval as eval
import model.predict as predict
import utils.DataManager
from distutils.dir_util import copy_tree

class DeepResidualLearning():
    def __init__(self):
        relationlist1=[]
        pass

    @classmethod
    def read_dataset(self, input_file, *args, **kwargs):
        data,self.relationlist1=conversionutil1.Common_to_NYT(input_file)
        print(data)
        return data

    @classmethod
    def train(self, dataobject, *args, **kwargs):
        print("Model is training..")
        train.train(dataobject,self.relationlist1,**kwargs)
        pass

    @classmethod
    def predict(self, testfile,entity_1=None, entity_2=None, trained_model=None, *args, **kwargs):
        outputfile=predict.predict(testfile,self.relationlist1,**kwargs)
        return outputfile

    @classmethod
    def evaluate(self, testfile,trained_model=None, *args, **kwargs):
        eval.eval(testfile,self.relationlist1,**kwargs)
        pass

    def save_model(self,model_path,**kwargs):
        fromDirectory = kwargs.get('from_dir',"runs/model_output/checkpoints")
        toDirectory = model_path
        copy_tree(fromDirectory, toDirectory)
        print("Saved model to",toDirectory)
        pass

    def load_model(self,**kwargs):
        path=kwargs.get('model_path',"runs/model_output/checkpoints/")
        sess,graph=eval.load_model(model_path=path)
        print(sess,graph)
        pass