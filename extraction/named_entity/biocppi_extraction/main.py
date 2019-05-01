
import sys
from biocppi_utils import copy_predictions_to_predictions_with_header, load_groundTruth_from_predictions

from biocppi_extraction import biocppi_extraction


def main(inputFilePath):
    #instantiate a model!

    # test params:
    test_params = {'num_ensembles':2,'num_iterations':101,'num_it_per_ckpt':1000000}  # note, it num_it_per_ckpt > num_iterations then num_it_per_ckpt will be set to half of num_iterations
    biocppi = biocppi_extraction(**test_params)

    # convert dataset to properformat used by training
    # 1] read_dataset()

    # test unittest...good!
    dataset_name = 'unittest'
    file_dict = {'train':{'data':inputFilePath},'dev':{},'test':{}}

    # # test conll2003...good!
    # dataset_name = 'CoNLL_2003'
    # # dataset_dir = '/Users/olderhorselover/USC/spring2019/csci_548_diotw/project/groupedProject/conll2003_corpus/smol/'  # smol sample
    # dataset_dir = '/Users/olderhorselover/USC/spring2019/csci_548_diotw/project/groupedProject/conll2003_corpus/'
    # raw_data_train_file = dataset_dir + 'train.txt'
    # raw_data_dev_file = dataset_dir + 'dev.txt'
    # raw_data_test_file = dataset_dir + 'test.txt'
    # file_dict = {'train':{'data':raw_data_train_file},'dev':{'data':raw_data_dev_file},'test':{'data':raw_data_test_file}}

    # # test ontoNotes5.0...good!
    # dataset_name = 'OntoNotes_5p0'
    # # dataset_dir = '/Users/olderhorselover/USC/spring2019/csci_548_diotw/project/groupedProject/ontoNotes5_corpus/OntoNotes-5.0-NER-BIO/smol/'  # smol sample
    # dataset_dir = '/Users/olderhorselover/USC/spring2019/csci_548_diotw/project/groupedProject/ontoNotes5_corpus/OntoNotes-5.0-NER-BIO/'
    # raw_data_train_file = dataset_dir + 'onto.train.ner'
    # raw_data_dev_file = dataset_dir + 'onto.development.ner'
    # raw_data_test_file = dataset_dir + 'onto.test.ner'
    # file_dict = {'train':{'data':raw_data_train_file},'dev':{'data':raw_data_dev_file},'test':{'data':raw_data_test_file}}

    # # test CHEMDNER...good!
    # dataset_name = 'CHEMDNER'
    # # dataset_dir = '/smol/'  # smol sample
    # dataset_dir = '/Users/olderhorselover/USC/spring2019/csci_548_diotw/project/groupedProject/chemdner_corpus/'
    # raw_data_train_file = dataset_dir + 'training.abstracts.txt'
    # raw_data_dev_file = dataset_dir + 'development.abstracts.txt'
    # raw_data_test_file = dataset_dir + 'evaluation.abstracts.txt'
    # raw_annot_train_file = dataset_dir + 'training.annotations.txt'
    # raw_annot_dev_file = dataset_dir + 'development.annotations.txt'
    # raw_annot_test_file = dataset_dir + 'evaluation.annotations.txt'
    # file_dict = {'train':{'data':raw_data_train_file,'extra':raw_annot_train_file},
    #             'dev':{'data':raw_data_dev_file,'extra':raw_annot_dev_file},
    #             'test':{'data':raw_data_test_file,'extra':raw_annot_test_file}}

    
    data = biocppi.read_dataset(file_dict, dataset_name)  # data read, converted, and written to files in proper location expected by train
    
    # train model
    #data = []  # implementation
    data_train = data['train']  # test passing actual data [empty also works]
    biocppi.train(data_train)
    print('DONE TRAIN')
    sys.exit()

    # predict using trained model
    data_test = data['test']
    predictions = biocppi.predict(data_test)  # test passing actual data [empty also works]
    print(len(predictions))

    
    outputPredictionsFile = 'predictions.txt'
    finalOutputFile = copy_predictions_to_predictions_with_header(raw_predictions_filename=outputPredictionsFile)
    
    # read from predictions file for evaluate
    evaluation_results = biocppi.evaluate(None,None)

    # use direct data for evaluate
    # groundTruths = load_groundTruth_from_predictions(raw_predictions_filename=outputPredictionsFile)
    # evaluation_results = biocppi.evaluate(predictions,groundTruths)

    print('%s'%str(evaluation_results))

    return finalOutputFile  # NOT FULLY IMPLEMENTED



if __name__=='__main__':

    inputFilePath='test/sample_input.txt'

    main(inputFilePath)

