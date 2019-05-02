import parent
import data_helpers
import train
import predict
import eval
import conversionutil1
import conversionutil2
from distutils.dir_util import copy_tree
from parent import RelationExtraction

class RelationExtractionCNN(RelationExtraction):

    def __init__(self):
        pass

    @classmethod
    def read_dataset(self, input_file, *args, **kwargs):
        data = conversionutil1.Common_to_SemEval(input_file)
        return data

    @classmethod
    def train(self, dataobject, *args, **kwargs):
        print("Model is training..")
        train.train(dataobject,**kwargs)
        pass

    @classmethod
    def predict(self, dataobject, entity_1=None, entity_2=None, trained_model=None, *args, **kwargs):
        outputfile=predict.predict(dataobject,**kwargs)
        return outputfile

    @classmethod
    def evaluate(self, dataobject, trained_model=None, *args, **kwargs):
        eval.eval(dataobject,**kwargs)
        pass

    def save_model(self,model_path,**kwargs):
        fromDirectory = kwargs.get('from_dir',"runs/model_checkpoint/checkpoints")
        toDirectory = model_path
        copy_tree(fromDirectory, toDirectory)
        print("Saved model to",toDirectory)
        pass

    def load_model(self,**kwargs):
        path=kwargs.get('model_path',"runs/model_checkpoint/checkpoints/")
        sess,graph=eval.load_model(model_path=path)
        print(sess,graph)
        pass
    
    @classmethod
    #not used method
    def data_preprocess(self, input_data, *args, **kwargs):
        pass
    
    #not used method
    @classmethod
    def tokenize(self, input_data, ngram_size=None, *args, **kwargs):
        pass