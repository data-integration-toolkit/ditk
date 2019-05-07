from cnn_blstm import CNN_BLSTM
from ner import Ner
from validation import compute_f1
from keras.optimizers import Nadam

class few_shot(Ner):

    def __init__(self, epochs=30, dropout=0.5, dropout_recurrent=0.25, LSTM_state_size=200,
                 conv_size=3, learning_rate=0.0105, optimizer=Nadam()):
        self.EPOCHS = epochs
        self.DROPOUT = dropout
        self.DROPOUT_RECURRENT = dropout_recurrent
        self.LSTM_STATE_SIZE = LSTM_state_size
        self.CONV_SIZE = conv_size
        self.LEARNING_RATE = learning_rate
        self.OPTIMIZER = optimizer

        self.cnnBLSTM_Obj = CNN_BLSTM(self.EPOCHS, self.DROPOUT, self.DROPOUT_RECURRENT, self.LSTM_STATE_SIZE,
                                      self.CONV_SIZE, self.LEARNING_RATE, self.OPTIMIZER)

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
        pass


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

        train_data = file_dict['train']['train']
        dev_data = file_dict['dev']['dev']
        test_data = file_dict['test']['test']

        self.cnnBLSTM_Obj.loadData(train_data, dev_data, test_data)

        # Datasets are stored in cnnBLSTM_Obj as instance variables

    def train(self, data, *args, **kwargs):
        """
        args
            Will not be using any arguments as the train() function in my code takes no arguments.
            CNN_BLSTM object has values stored as instance variables

        Return
            None. The weights and other parameters are stored as instance variables of CNN_BLSTM
        """

        """ Preprocess Data Before Training """

        self.cnnBLSTM_Obj.addCharInfo()
        self.cnnBLSTM_Obj.embed()
        self.cnnBLSTM_Obj.createBatches()
        self.cnnBLSTM_Obj.buildModel()

        self.cnnBLSTM_Obj.train()

    def predict(self, data, *args, **kwargs):
        """
        :param data: data for predictions
        :param args: None
        :param kwargs: None
        :return: predictions --> List of predicted labels
        """

        predictions, truthLables = self.cnnBLSTM_Obj.tag_dataset(data, self.cnnBLSTM_Obj.model)

        # Save predictions and truthLabels
        self.predictions = predictions
        self.groundTruthLabels = truthLables

        return predictions

    def evaluate(self, predictions, groundTruths, *args, **kwargs):
        """
        :param predictions:     Predictions from predict() function
        :param groundTruths:    ground Truth values
        :param args:            None
        :param kwargs:          None
        :return:                A tuple with precision, recall and F1 scores
        """

        precision, recall, f1 = compute_f1(predictions, groundTruths, self.cnnBLSTM_Obj.idx2Label)

        return (precision, recall, f1)


    def save_model(self, file):
        """
        :param file: Where to save the model - Optional function
        :return:
        """
        pass


    def load_model(self, file):
        """
        :param file: From where to load the model - Optional function
        :return:
        """
        pass


def main():
    few_shot_obj = few_shot(epochs=2)

    file_dict = {'train':{'train':"data/train.txt"},
                 'dev':{'dev':"data/dev.txt"},
                 'test':{'test':"data/test.txt"}}

    # Load datasets
    few_shot_obj.read_dataset(file_dict, "")

    # Training
    few_shot_obj.train(None)

    # Predictions
    predictions = few_shot_obj.predict(few_shot_obj.cnnBLSTM_Obj.test_batch)

    # Run Evaluations
    evaluations = few_shot_obj.evaluate(predictions, few_shot_obj.groundTruthLabels)
    print ("Evaluation metrics:")
    print("Precision:", evaluations[0] * 100, "\tRecall:", evaluations[1] * 100, "\tF1 score:", evaluations[2] * 100)
