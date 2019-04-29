'''
Group 6. NER
Paper: "Robust Lexical Features for Improved Neural Network Named-Entity Recognition"

parent's README.md URL: https://github.com/AbelJJohn/csci548sp19projectner
* Benchmarks(3): Plain text
+ CoNLL 2003: work with it
+ OntoNotes: work with it
- CHEMDNER: CANNOT work with it because it is bio data

* Metrics(3): work with ALL
- Recall
- Precession
- F1
'''

# import ditk # super class????????
import preprocess

# class NER_with_LS(ditk.NER):
class NER_with_LS():
    '''
    TODO:
     - 2 datasets are avilable; conll | ontonotes
    '''
    def read_dataset(self, file_dict, dataset_name) :
        """
        Reads a dataset in preparation for train or test. Returns data in proper format for train or test.
        Args:
            file_names: List of files representing the dataset to read. Each element is str, representing
                filename [possibly with filepath]
        Returns:
        	train & text data: dictionary which has np.array type of values
        """

        print("====== 2. Getting data ======") 
        data = preprocess.read_data(dataset_name)

        print(" TMEP :::: Let's see the raw_data dictionary !! ")
        for k, v in raw_data.items():
            print(k)

        return data


    '''
    TODO:
     - extra lexical embeddings(ls_embedding.py and .json) files are required to generate LS embeddings for any word
    '''
    def train(self, train_data, *args, **kwargs):
        """
        Args:
        	type of train_data is np.array (or could be list)

        Returns:
        	None. Trained model stored internally to class instance state
        """
        pass


    '''
    TODO:
     - predict function should be extracted from NER_with_LSmodel.py 
     - len(predictions) should equal len(data) AND the ordering should not change [important for evalutation]
    '''
    def predict(self, data, *args, **kwargs):
        """
        Args:
            data: tuple/dic which has np.array as values OR just 1 np.array as input_data

        Returns:
            predictions_label: list of tuple
                Each tuple is (start index, span, mention text, mention type)
                Where:
                 - start index: int, the index of the first character of the mention span. None if not applicable.
                 - span: int, the length of the mention. None if not applicable.
                 - mention text: str, the actual text that was identified as a named entity. Required.
                 - mention type: str, the entity/mention type. None if not applicable.
        """

        # return predictions
        pass

    '''
    TODO:
     - this evaluate is more like get score function
     - get the score for each metrics
    '''
    def evaluate(self, predictions, groundTruths, *args, **kwargs): 
        """
        Args:
            predictions: [tuple,...], list of tuples
            groundTruths: [tuple,...], list of tuples representing ground truth.

        Returns:
            metrics: tuple with (p,r,f1). Each element is float.

        Raises:
            None
        """
        # return (precision, recall, f1)
        pass

    '''
    TODO:
     - when I read ground_truth_label in read_dataset(); type is np.array, so ground_truth_label should match the format to predicted_label format which will be list.
    '''
    def convert_ground_truth(self, data, *args, **kwargs):
        """
        Args:
            data: data in proper [arbitrary] format for train or test. [i.e. format of output from read_dataset]

        Returns:
            ground_truth: list of tuples. [SAME format as output of predict()]
                Each tuple is (start index, span, mention text, mention type)
        """
        # return ground_truth
        pass



