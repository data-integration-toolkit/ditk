from ner_blstm_cnn import ner_blstm_cnn
import os

def main(train_file, dev_file=None, test_file=None):

    if dev_file==None and test_file==None:
        inputFiles = {'train': train_file,
                      'dev': train_file,
                      'test': train_file}
    else:
        inputFiles = {'train': train_file,
                      'dev': dev_file if dev_file!=None else train_file,
                      'test': test_file if test_file!=None else train_file}

    # instatiate the class
    ner = ner_blstm_cnn(5)

    # read in a dataset for training
    data = ner.read_dataset(inputFiles)

    # trains the model and stores model state in object properties or similar
    ner.train(data)

    # get ground truth from data for test set
    ground = ner.convert_ground_truth(data)

    # generate predictions on test
    predictions = ner.predict(data)

    # calculate Precision, Recall, F1
    P,R,F1 = ner.evaluate(predictions, ground)

    print('Precision: %s, Recall: %s, F1: %s'%(P,R,F1))

    output_file = os.path.dirname(train_file)
    output_file_path = os.path.join(output_file, "output.txt")

    with open(output_file_path, 'w') as f:
        for index, (g, p) in enumerate(zip(ground, predictions)):
            if len(g[3])==0:
                f.write("\n")
            else:
                f.write(g[2] + " " + g[3] + " " + p[3] + "\n")

    return output_file_path



if __name__ == "__main__":
    train_file = "/Users/lakshya/Desktop/CSCI-548/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs-master/conll/train.txt"
    dev_file = "/Users/lakshya/Desktop/CSCI-548/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs-master/conll/valid.txt"
    test_file = "/Users/lakshya/Desktop/CSCI-548/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs-master/conll/test.txt"
    main(train_file, dev_file, test_file)