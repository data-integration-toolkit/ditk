from distutils.dir_util import copy_tree

from src.parent_relation_extraction import RelationExtraction
from utils_data import SemEval2010_util as data_converter
from src import eval as evaluation_object, train as training_object


class AttentionBasedBiLstmModel(RelationExtraction):


    @classmethod
    def read_dataset(self, input_file, *args, **kwargs):
        """
        Reads the dataset from input file
        :param input_file:
        :param args: not applicable
        :param kwargs:
        :return:
        """

        print("Reading input file")

        dataType = kwargs.get("dataType","common");

        data = []
        if(dataType == "common"):

            lines = [line.strip() for line in open(input_file)]
            for line in lines:
                data.append(line.split('\t'))



        return data

    @classmethod
    def data_preprocess(self, input_data, *args, **kwargs):

        #method not used

        """
        takes data -> reformats it if needed for the required model
        (for eg : for the datasets which don't belong to my model)

        takes data from the input file and preprocess it to get text
        data and labels of the same

        :param input_data:
        :param args: not applicable
        :param kwargs:
        :return: x_train - text data for training
                y - labels for the above text data
        """
        pass

    @classmethod
    def tokenize(self, input_data, ngram_size=None, *args, **kwargs):

        #method not used

        """
        :param input_data:
        :param ngram_size:
        :param args:
        :param kwargs:
        :return:
        """
        pass

    @classmethod
    def train(self, train_data,*args, **kwargs):


        """
           Compulsory Arguments given through kwargs :
           embedding_file_path : Path of the file ex: path of the glove embedding file file

           Optional Arguments - if not given default will be used

           max_sentence_length : Max sentence length in data
           dev_sample_percentage : Percentage of the training data to use for validation

           'for embedding'
           embedding_dim : Dimensionality of word embedding
           emb_dropout_keep_prob : Dropout keep probability of embedding layer

           'AttLSTN'
           hidden_size : Dimensionality of RNN hidden
           rnn_dropout_keep_prob : Dropout keep probability of RNN


           dropout_keep_prob : Dropout keep probability of output layer
           l2_reg_lambda :L2 regularization lambda

           'Training Parameters'
           batch_size:Batch Size
           num_epochs :Number of training epochs
           display_every :Number of iterations to display training information
           evaluate_every : Evaluate model on dev set after this many steps
           num_checkpoints: Number of checkpoints to store
           learning_rate : Which learning rate to start with

           :param train_data:  train_data is a form of x, y where x is the text data and y is the corresponding labels.
           :param args:
           :param kwargs:
           :return:  None : makes checkpoints to the model and model

           """

        sem_eval_data_format = data_converter.Common_to_SemEval2010(train_data,'train')

        print("Training Model with below paramters..................")
        """
        Printing values given to training model
        """

        if('embedding_path' not in kwargs):
            raise ValueError("Please provide a embedding path")

        print("Max Sentence Length:- ",kwargs.get('max_sentence_length',90))
        print("Dev Sample Percentage:=- ",kwargs.get('dev_sample_percentage',0.1))
        print("Embedding Path:- ",kwargs.get('embedding_path','../res/glove.6B.100d.txt'))
        print("Embedding Dimensions:- ",kwargs.get('embedding_dim',100))
        print("Dropout probability of embedding layer:- ",kwargs.get('emb_dropout_keep_prob',0.7))
        print("Dimensionality of RNN hidden:- ",kwargs.get('hidden_size',100))
        print("Dropout probability of RNN:- ",kwargs.get('rnn_dropout_keep_prob',0.7))
        print("L2 Regularization lamba:- ",kwargs.get('l2_reg_lambda','1e-5'))
        print("Batch Size:- ",kwargs.get('batch_size',10))
        print("Num Epochs:- ",kwargs.get('num_epochs',100))
        print("Display Every:- ",kwargs.get('display_every',10))
        print("Evaluate Every:- ",kwargs.get('evaluate_every',100))
        print("Number of Checkpoints:- ",kwargs.get('num_checkpoints',5))
        print("Learning Rate:- ",kwargs.get('learning_rate',1.0))
        print("Decay Rate:- ",kwargs.get('decay_rate',0.9))

        print("\n\nTraining Model..................")

        training_object.train(sem_eval_data_format, **kwargs)

        print("\n\nTraining Completed.................")

        pass

    @classmethod
    def predict(self, test_data, entity_1=None, entity_2=None, trained_model=None, *args, **kwargs):

        sem_eval_data_predict = data_converter.Common_to_SemEval2010(test_data,'predict')

        predictions,output_file_path = evaluation_object.predict(sem_eval_data_predict,**kwargs)

        """
         Compulsory Arguments given through kwargs:
         checkpoint_dir: Checkpoint directory from training run (give path of checkpoint)


        test data will be formatted first to the required format and then give to the predict function

        data will have entities mentioned and predict will detect the relationship
        As the entities are defined first , my program only returns the relationship between them and not
        the names of the entity.

        :param test_data:
        :param entity_1: not applicable
        :param entity_2: ot applicable
        :param trained_model: not applicable
        :param args:
        :param kwargs:
        :return:  tuple (Entity,Relation,Entity)
        """
        return predictions,output_file_path

    @classmethod
    def evaluate(self, input_data, trained_model=None, *args, **kwargs):

        print("Evaluating Model...............")

        sem_eval_data_predict = data_converter.Common_to_SemEval2010(input_data,'test')

        evaluation_object.eval(sem_eval_data_predict,**kwargs)

        """
        Compulsory Arguments given through kwargs
        checkpoint_dir: Checkpoint directory from training run (give path of checkpoint)

        input_data : after reading test file from readdataset() and preprocessing
        it using datapreprocessing() it will be passed here. Evaluation will run on this data

        :param input_data:
        :param trained_model:
        :param args:
        :param kwargs:
        :return: (precision, recall, f1)
        """
        pass

    @classmethod
    def save_model(self,filePath,**kwargs):

        fromDirectory = kwargs.get('checkpoint_dir', "../runs/models/")
        toDirectory = filePath
        copy_tree(fromDirectory, toDirectory)
        print("Saved model to", toDirectory)

        pass

    @classmethod
    def load_model(self,filePath,**kwargs):

        tensorFlow_session,vocab,tensorFlow_graph = evaluation_object.restoreModel(filePath+'/checkpoints',checkpoint_dir='../runs/models/checkpoints')

        print(tensorFlow_session)
        print(vocab)
        print(tensorFlow_graph)

        return tensorFlow_graph,tensorFlow_session,vocab

    def main(self,input_file):


        common_format_data = AttentionBasedBiLstmModel.read_dataset(input_file)

        print(common_format_data)

        AttentionBasedBiLstmModel.train(common_format_data, embedding_path="../res/glove.6B.100d.txt")


        test_common_format_data = AttentionBasedBiLstmModel.read_dataset(input_file)

        AttentionBasedBiLstmModel.evaluate(test_common_format_data, checkpoint_dir="../runs/models/checkpoints")

        predictions,output_file_path = AttentionBasedBiLstmModel.predict(test_common_format_data,
                                                        checkpoint_dir="../runs/models/checkpoints")
        print(predictions)

        file_save_model = "../saved_model_dir"

        AttentionBasedBiLstmModel.save_model(file_save_model)

        AttentionBasedBiLstmModel.load_model(file_save_model)


        return output_file_path