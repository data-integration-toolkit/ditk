
from drtrnn_utils import copy_predictions_to_predictions_with_header 

from disease_name_recognition_through_rnn import disease_name_recognition_through_rnn


def main(inputFilePath):
    #instantiate a model!

    # test params:
    test_params = {'n_iter':501,'m_report':100,'save_checkpoint_files':False}
    drtrnn = disease_name_recognition_through_rnn(**test_params)

    # print('input file: %s'%inputFilePath)

    # print(type(myModel))

    # convert dataset to properformat used by training
    # 1] read_dataset()
    file_dict = {'train':{'data':inputFilePath},'dev':{},'test':{}}
    dataset_name = 'unittest'
    data = drtrnn.read_dataset(file_dict, dataset_name)  # data read, converted, and written to files in proper location expected by train
    # 2] intermediate step, generate *_tags files, *_words files, vocab file

    # train model
    #data = []  # implementation
    data_train = data['train']  # test passing actual data [empty also works]
    drtrnn.train(data_train)

    # predict using trained model
    data_test = data['test']
    drtrnn.predict(data_test)  # test passing actual data [empty also works]

    
    outputPredictionsFile = 'predictions.txt'
    finalOutputFile = copy_predictions_to_predictions_with_header(raw_predictions_filename=outputPredictionsFile)
    
    return finalOutputFile  # NOT FULLY IMPLEMENTED



if __name__=='__main__':

    inputFilePath='test/sample_input.txt'

    main(inputFilePath)

