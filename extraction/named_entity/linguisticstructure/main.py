from linguisticstructure_ner import LingusticStructureforNER
import os

def main(input_file):
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    ner = LingusticStructureforNER()

    embeddings = {}
    senna_hash_path = 'tests'
    glove_path = 'tests/test_embeddings.txt'
    embeddings['glove_path'] = glove_path
    embeddings['senna_hash_path'] = senna_hash_path
    file_dict = {"train": input_file, "test": input_file, "dev": input_file}

    data = ner.read_dataset(file_dict, embeddings=embeddings)
    ner.train(data, max_epoches=1)
    predictions = ner.predict(data['test'])

    actual_tags = ner.convert_ground_truth(data['test'])

    precision, recall, f1 = ner.evaluate(predictions, actual_tags)
    file_location = os.path.join(os.getcwd(),'ner_test_output.txt')
    return file_location
