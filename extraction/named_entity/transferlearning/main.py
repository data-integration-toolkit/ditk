from transferlearning_ner import TransferLearningforNER

import os

def main(input_file):
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    ner = TransferLearningforNER()
    embedding_file= ''


    src_data = ner.read_dataset(file_dict={"train": input_file, "test": input_file, "dev": input_file}, dataset_name='src')
    tgt_data = ner.read_dataset(file_dict={"train": input_file, "test": input_file, "dev": input_file}, dataset_name='tgt')
    ner.train(tgt_data, src_data=src_data, classifier='CRF', transfer_method='tgt')
    preds = ner.predict(tgt_data['test'])
    actual_tags = ner.convert_ground_truth(tgt_data['test'])
    scores = ner.evaluate(preds, actual_tags)
    file = os.path.join(os.getcwd(),'ner_test_output.txt')
    return file
