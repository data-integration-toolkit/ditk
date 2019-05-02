#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import abc

class Blocking(abc.ABC):

    # Any shared data strcutures or methods should be defined as part of the parent class.
    
    # A list of shared arguments should be defined for each of the following methods and replace (or precede) *args.
    
    # The output of each of the following methods should be defined clearly and shared between all methods implemented by members of the group. 
    
    def __init__(self, blocking_module,**kargs):
        '''
        Initialise object
        Args:
            blocking_module: String that mentions the blocking module being used.
                        **kargs: Dictionary containing parameters needed in blocking pipeline. 
        '''
        self.blocking_module = blocking_module

    @classmethod
    @abc.abstractmethod
    def read_dataset(filepath_list, *args, **kwargs):
        '''
        Accepts list of URI's and returns a List of Pandas DataFrames.
        Args:
            filepath_list: List of string path to the dataset.
            *args: Additional arguments can be mentioned.
                        **kargs: Parameters for reading datasets.

        Returns: List of Pandas DataFrames
        '''
        pass

    @classmethod
    @abc.abstractmethod
    def train(dataframe, *args, **kwargs):
        '''
        Accepts Pandas Dataframe to train the model on and returns the trained model
        Args:
            dataframe: Pandas dataframe with training vectors under 'vectors' column. Row id's 
                            will be used as node id's in the tree built by the related pairs algorithms. 
            *args: Empty
                        **kargs: Dictionary containing requisite parameters for training/building the 
                            algorithm specified in their original papers/githubs. 

        Returns: Trained model
        '''
        pass

    @classmethod
    @abc.abstractmethod
    def predict(dataframe_list, *args, **kwargs):
        '''
        Given a trained model and Pandas Dataframe of datasets, predict() returns a list of pairs of
            related ids as a list of tuples.
        Args:
            model: tensorflow model used to predict.
            dataframe_list: List of Pandas Dataframes.
            *args: Additional arguments.
                        **kargs: Parameters requisite for prediciting related ids as defined by the original papers/githubs. 

        Returns: List of pairs of elements related in the dataset. Lists of Lists of tuples [[(id_0,0.99),(id_3,0.95)],...,[(id_5,0.97),(id_1,0.83)]].
        '''
        pass

    @classmethod
    @abc.abstractmethod
    def evaluate(model,groundtruth, dataframe_list, *args, **kwargs):
        '''
        Given the ground truth and list of dataframes to predict the related ids for, evaluate() returns the Precision,
            Recall and Reduction Ratio metrics.
        Args:
            model: Model to evaluate
            groundtruth: String path to or Dataframe of the ground truth data.
            dataframe_list: List of Pandas Dataframes with the data to predict the related id's for. 
            *args: If more than 1 dataset, string dataset path can be mentioned as additional arguments.
            **kargs: Parameters requisite for prediciting related ids as defined by the original papers/githubs.

        Returns: Precision, Recall, Reduction_Ratio
        '''
        pass

    @classmethod
    @abc.abstractmethod
    def load_model(path, *args, **kwargs):
        '''
        Given a path and requisit parameters to load model will store model as a class member variable.
        Args:
            path: Path to saved model
            *args: None
            **kargs: Parameters requisite for loading models defined by the original papers/githubs.

        Returns: None
        '''
        pass
    @classmethod
    @abc.abstractmethod
    def save_model(path, *args, **kwargs):
        '''
        Given a path and requisit parameters to save model will store the member variable model to path.
        Args:
            path: Path to save model to
            *args: None
            **kargs: Parameters requisite for saving models defined by the original papers/githubs.

        Returns: None
        '''
        pass


# In[ ]:




