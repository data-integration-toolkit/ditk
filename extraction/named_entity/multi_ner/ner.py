import abc

class Ner(abc.ABC):

    @abc.abstractmethod
    def convert_ground_truth(self, data, *args, **kwargs):  # <--- implemented PER class
        """
        Converts test data into common format for evaluation [i.e. same format as predict()]
        This added step/layer of abstraction is required due to the refactoring of read_dataset_traint()
        and read_dataset_test() back to the single method of read_dataset() along with the requirement on
        the format of the output of predict() and therefore the input format requirement of evaluate(). Since
        individuals will implement their own format of data from read_dataset(), this is the layer that
        will convert to proper format for evaluate().
        Args:
            data: data in proper [arbitrary] format for train or test. [i.e. format of output from read_dataset]
        Returns:
            ground_truth: [tuple,...], i.e. list of tuples. [SAME format as output of predict()]
                Each tuple is (start index, span, mention text, mention type)
                Where:
                 - start index: int, the index of the first character of the mention span. None if not applicable.
                 - span: int, the length of the mention. None if not applicable.
                 - mention text: str, the actual text that was identified as a named entity. Required.
                 - mention type: str, the entity/mention type. None if not applicable.
        Raises:
            None
        """
        # IMPLEMENT CONVERSION. STRICT OUTPUT FORMAT REQUIRED.

        # return ground_truth

    @abc.abstractmethod
    def read_dataset(self, file_dict, dataset_name, *args, **kwargs):  # <--- implemented PER class
        """
        Reads a dataset in preparation for train or test. Returns data in proper format for train or test.
        Args:
            file_dict: dictionary
                 {
                    "train": dict, {key="file description":value="file location"},
                    "dev" : dict, {key="file description":value="file location"},
                    "test" : dict, {key="file description":value="file location"},
                 }
            dataset_name: str
                Name of the dataset required for calling appropriate utils, converters
        Returns:
            data: data in arbitrary format for train or test.
        Raises:
            None
        """
        # IMPLEMENT READING
        # pass

    @abc.abstractmethod
    def train(self, data, *args, **kwargs):  # <--- implemented PER class
        """
        Trains a model on the given input data
        Args:
            data: iterable of arbitrary format. represents the data instances and features and labels you use to train your model.
        Returns:
            ret: None. Trained model stored internally to class instance state.
        Raises:
            None
        """
        # IMPLEMENT TRAINING.
        # pass

    @abc.abstractmethod
    def predict(self, data, *args, **kwargs):  # <--- implemented PER class WITH requirement on OUTPUT format!
        """
        Predicts on the given input data. Assumes model has been trained with train()
        Args:
            data: iterable of arbitrary format. represents the data instances and features you use to make predictions
                Note that prediction requires trained model. Precondition that class instance already stores trained model
                information.
        Returns:
            predictions: [tuple,...], i.e. list of tuples.
                Each tuple is (start index, span, mention text, mention type)
                Where:
                 # - start index: int, the index of the first character of the mention span. None if not applicable.
                 # - span: int, the length of the mention. None if not applicable.
                 # - mention text: str, the actual text that was identified as a named entity. Required.
                 # - mention type: str, the entity/mention type. None if not applicable.
                 NOTE: len(predictions) should equal len(data) AND the ordering should not change [important for
                     evalutation. See note in evaluate() about parallel arrays.]
        Raises:
            None
        """
        # IMPLEMENT PREDICTION. STRICT OUTPUT FORMAT REQUIRED.

        # return predictions

    @abc.abstractmethod
    def evaluate(self, predictions, groundTruths, *args,
                 **kwargs):  # <--- common ACROSS ALL classes. Requirement that INPUT format uses output from predict()!
        """
        Calculates evaluation metrics on chosen benchmark dataset [Precision,Recall,F1, or others...]
        Args:
            predictions: [tuple,...], list of tuples [same format as output from predict]
            groundTruths: [tuple,...], list of tuples representing ground truth.
        Returns:
            metrics: tuple with (p,r,f1). Each element is float.
        Raises:
            None
        """
        # pseudo-implementation
        # we have a set of predictions and a set of ground truth data.
        # calculate true positive, false positive, and false negative
        # calculate Precision = tp/(tp+fp)
        # calculate Recall = tp/(tp+fn)
        # calculate F1 using precision and recall

        # return (precision, recall, f1)

        	
    @abc.abstractmethod
    def save_model(self, file):
        """
        :param file: Where to save the model - Optional function
        :return:
        """
        pass

    @abc.abstractmethod
    def load_model(self, file):
        """
        :param file: From where to load the model - Optional function
        :return:
        """
        pass
        

"""
# Sample workflow:
file_dict = {
                "train": {"data" : "/home/sample_train.txt"},
                "dev": {"data" : "/home/sample_dev.txt"},
                "test": {"data" : "/home/sample_test.txt"},
             }
dataset_name = 'CONLL2003'
# instatiate the class
myModel = myClass() 
# read in a dataset for training
data = myModel.read_dataset(file_dict, dataset_name)  
myModel.train(data)  # trains the model and stores model state in object properties or similar
predictions = myModel.predict(data['test'])  # generate predictions! output format will be same for everyone
test_labels = myModel.convert_ground_truth(data['test'])  <-- need ground truth labels need to be in same format as predictions!
P,R,F1 = myModel.evaluate(predictions, test_labels)  # calculate Precision, Recall, F1
print('Precision: %s, Recall: %s, F1: %s'%(P,R,F1))
"""