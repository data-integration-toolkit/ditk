import os
from twitterner import TwitterNER
def main(input_file_path):
    main_dir_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
    train_files = ["./data/cleaned/train.BIEOU.tsv"]
    dev_files = ["./data/cleaned/dev.BIEOU.tsv"]
    test_files = ["./data/cleaned/test.BIEOU.tsv"]
    vocab_file = "./vocab.no_extras.txt"
    outdir = "./test_exp"
    file_dict = {'train':train_files,'dev':dev_files, 'test':test_files}
    # wordvec_file = "/home/entity/Downloads/GloVe/glove.twitter.27B.200d.txt.processed.txt"
    wordvec_file = "./data/glove.twitter.27B/glove.twitter.27B.200d.txt"
    wordvec_file_processed = "./data/glove.twitter.27B/glove.twitter.27B.200d.txt.processed.txt"
    dictionary_dir="./data/cleaned/custom_lexicons/"
    gimple_twitter_brown_clusters_dir="./50mpaths2"
    data_brown_cluster_dir="brown_clusters_wnut_and_hege/"
    data_clark_cluster_dir="clark_clusters_wnut_and_hege/"
    enriched_brown_cluster_dir = main_dir_path + "/brown_clusters_wnut_and_hege"
    
    enriched_clark_cluster_dir = main_dir_path+ "/clark_clusters_wnut_and_hege"
    twitterner = TwitterNER()
    twitterner.read_dataset(outdir,file_dict, vocab_file, False, dataset='WNUT')
    #twitterner.train()
    results = twitterner.predict(input_file_path)
    print results
    twitterner.evaluate()
    output_path = main_dir_path + './output.txt'
    return output_path

if __name__ == '__main__':
    input_file_path = '/home/khadutz95/TwitterNER/NoisyNLP/tests/Ner_test_input.txt'
    output_file_path = main(input_file_path)
    print output_file_path
