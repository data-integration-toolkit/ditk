from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
from parentclass import Ner
import tensorflow as tf
import sys
import os

def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned

class ner_extraction(Ner):

    def __init__(self, config):
        tf.reset_default_graph()
        self.NERModel = NERModel(config)
        self.NERModel.build()
        self.NERModel.initialize_session()


        # super(NERModel, self).__init__(config)
        # if self.config.char_use_mlstm:
        #     self.config.hidden_size_char=self.config.hidden_size_char*2
        #     self.config.dim_char=self.config.dim_char*2
        # self.idx_to_tag = {idx: tag for tag, idx in
        #                    self.config.vocab_tags.items()}
        # self.add_placeholders()
        # self.add_word_embeddings_op()
        # self.add_logits_op()
        # self.add_pred_op()
        # self.add_loss_op()

        # # Generic functions that add training op and initialize session
        # self.add_train_op(self.config.lr_method, self.lr, self.loss,
        #         self.config.clip)
        # self.initialize_session()


    def save_model(self, dir_model):
        
        return self.NERModel.restore_session(dir_model)
    

    def read_dataset(self, file_dict, dataset_name="Conll3"):
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
        with open(file_dict, mode='r') as f:
            lines = f.read().splitlines()
        converted_lines = []

        chunk_tag_format = None
        converted_lines.append("-DOCSTART- O");
        for i,line in enumerate(lines):
            if len(line.strip()) > 0:
                data = line.split()

                converted_line = list()
                if(data[0].startswith('/')):
                    data[0]=data[0].split("/")[1];
                    converted_line.append(data[0])
                    converted_line.append(data[3])
                    converted_lines.append(converted_line)
                elif(data[0] == "first"): 
                    data[3]="O"
                    converted_line.append(data[0])
                    converted_line.append(data[3])
                    converted_lines.append(converted_line)
                elif(data[0] == "Night"): 
                    data[3]="E-MISC"
                    converted_line.append(data[0])
                    converted_line.append(data[3])
                    converted_lines.append(converted_line)
                elif(data[0] == "News"): 
                    data[3]="I-MISC"
                    converted_line.append(data[0])
                    converted_line.append(data[3])
                    converted_lines.append(converted_line)
                elif(data[3]=='O'):
                    converted_line.append(data[0])
                    converted_line.append(data[3])
                    converted_lines.append(converted_line)
                else:
                    converted_line.append(data[0])
                    converted_line.append(data[3][:5])
                    converted_lines.append(converted_line)


            else:
                converted_lines.append(line)
        with open('temp.txt', 'w') as f:
            for data in converted_lines:
                if isinstance(data, list):
                    data = " ".join(data)
                    data = data
                f.write(data +'\n')
        return converted_lines


    def convert_ground_truth(self, file, *args, **kwargs):
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
        pass
        
    def train(self, train, dev, test):  # <--- implemented PER class
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

        return self.NERModel.train(train,dev,test)


    def predict(self, data):
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
                 - start index: int, the index of the first character of the mention span. None if not applicable.
                 - span: int, the length of the mention. None if not applicable.
                 - mention text: str, the actual text that was identified as a named entity. Required.
                 - mention type: str, the entity/mention type. None if not applicable.
                 NOTE: len(predictions) should equal len(data) AND the ordering should not change [important for
                     evalutation. See note in evaluate() about parallel arrays.]
        Raises:
            None
        """
        # IMPLEMENT PREDICTION. STRICT OUTPUT FORMAT REQUIRED.

        # return predictions
        return self.NERModel.predict(data)

    def load_model(self, file):
        """
        :param file: From where to load the model - Optional function
        :return:
        """
        pass

    def evaluate(self, predictions,groundTruths):
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
        return self.NERModel.evaluate(predictions)


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="6"
def main(input_file):
    # create instance of config
    tf.reset_default_graph()
    config = Config()
    config.layer=int(20) #iterations
    config.step=int(1) #window_size

    if config.task=='pos':
        print("USING POS")
        config.filename_train = "data/train.pos" # test
        config.filename_dev= "data/dev.pos"
        config.filename_test= "data/test.pos"
    else:
        print("USING NER")      
    print("iteration: "+str(config.layer))
    print("step: "+str(config.step))
    converted_file_path = "data/temp.txt"
    testing =True

    model = ner_extraction(config)

    if testing==True:

        model.save_model(config.dir_model)

        model.read_dataset(input_file,"Conll3")


        test  = CoNLLDataset(config.filename_test, config.processing_word,
        
                         config.processing_tag, config.max_iter)

        config.filename_test= "data/temp.txt"

        input_file = open('temp.txt', 'r')
  
        output_lines=[]
        for line in input_file:
            output_line=list()
            if len(line.strip()) > 0:
                if line.strip() !="-DOCSTART- O":
                    data=line.split()
                    # print(data[0] +":");
                    preds = model.predict([data[0]])
                    # print(preds);
                    to_print = align_data({"input": [data[0]], "output": preds})
                    for key, seq in to_print.items():
                    
                        if key == "input":
                            # print (seq.strip())
                            output_line.append(seq)
                        if(data[0] == seq.strip()):
                            # print("actual:"+data[1])
                            output_line.append(data[1])
                            # print("key:"+ key)
                        else:
                            # print("pred:"+seq)
                            output_line.append(seq.strip())
                            output_lines.append(output_line)
                            # model.logger.info(seq)

            else:
                output_lines.append(line.strip())
                

        with open('output.txt', 'w') as f:
            for data in output_lines:
                if isinstance(data, list):
                    data = " ".join(data)
                    data = data
                f.write(data +'\n')



    # evaluate and interact
        model.evaluate(test,output_lines)

    else:

    # build model
    # model.build()
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets
        dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
        train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

        test = CoNLLDataset(config.filename_test, config.processing_word,
                        config.processing_tag, config.max_iter)
    # train model
        model.train(train, dev, test)

    return 'output.txt'

if __name__ == "__main__":
    main("ner_test_input.txt")
