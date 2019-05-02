import time

from code.evaluation.summarize_all_experiments import evaluation_main_modified
from code.model import config
from code.model.evaluate import evaluate_main_modified
from code.model.reader import parse_sequence_example
from code.model.train import train_modified
from code.model.util import load_train_args
from code.preprocessing.prepro_util import prepro_util_main
from entity_normalization import EntityNormalization
from code.preprocessing.prepro_aida import create_necessary_folders, process_aida, split_dev_test
from code.preprocessing.prepro_other_datasets import ProcessDataset, get_immediate_subdirectories
import os
import argparse
import tensorflow as tf

class End_to_end_neural_el(EntityNormalization):

    def __init__(self):
        pass


    def read_dataset(self, dataset_name: str, split_ratio: tuple, *args) -> tuple:
        '''
        :param dataset_name: path of dataset
        :param split_ratio: (train_ratio, validation_ration, test_ratio)
        :param kwargs: other parameters for specific model (optional)
        :return: train_data, valid_data, test_data
        '''

        dataset = tf.data.TFRecordDataset(dataset_name)
        dataset = dataset.map(parse_sequence_example)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(500)
        dataset = dataset.padded_batch(args['batch_size'], dataset.output_shapes)

        return dataset


    def train(self, train_set: list, args):
        '''
        :param train_set: a list of training data
        :return: trained model
        '''

        for each_training_set in train_set:
            train_modified(each_training_set, args)

        return


    def predict(self, model, test_set: list) -> list:
        '''
        :param model: a trained model
        :param test_set: a list of test data
        :return: a list of prediction, each item with the format
        (entity_name, wikipedia_url(optional), geolocation_url(optional), geolocation_boundary(optional))
        '''

        test_dataset = test_set[0]
        predictions = evaluate_main_modified(test_dataset)

        return predictions


    def evaluate(self, model, eval_set: list) -> tuple:
        '''
        :param model: a trained model
        :param eval_set: a list of validation data
        :return: (precision, recall, f1 score)
        '''
        base_folder, dev_set, test_set = eval_set[0], eval_set[1], eval_set[2]
        el_accuracy, ed_accuracy = evaluation_main_modified(base_folder, dev_set, test_set)
        return (el_accuracy[2], el_accuracy[3], el_accuracy[1])


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


def _parse_args_others(input_folder, output_folder, tokenizer_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--other_datasets_folder", default=input_folder)
    parser.add_argument("--output_folder", default=output_folder)
    parser.add_argument("--stanford_tokenizer_folder",
                        default=tokenizer_path)

    return parser.parse_args()


def _parse_args_aida(input_folder, output_folder):
    parser = argparse.ArgumentParser()
    parser.add_argument("--aida_folder", default=input_folder)
    parser.add_argument("--output_folder", default=output_folder)
    return parser.parse_args()


def _parse_args_train(args_dict):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default=args_dict['experiment_name'],  # "standard",
                        help="under folder data/tfrecords/")
    parser.add_argument("--training_name", default=args_dict['training_name'],
                        help="under folder data/tfrecords/")
    parser.add_argument("--shuffle_capacity", type=int, default=500)
    parser.add_argument("--debug", type=bool, default=False)

    parser.add_argument("--nepoch_no_imprv", type=int, default=args_dict['nepoch_no_imprv'])
    parser.add_argument("--improvement_threshold", type=float, default=0.3, help="if improvement less than this then"
                                                                                 "it is considered not significant and we have early stopping.")
    parser.add_argument("--clip", type=int, default=-1, help="if negative then no clipping")
    parser.add_argument("--lr_decay", type=float, default=-1.0, help="if negative then no decay")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_method", default="adam")
    parser.add_argument("--batch_size", type=int, default=args_dict['batch_size'])
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--train_ent_vecs", dest='train_ent_vecs', action='store_true')
    parser.add_argument("--no_train_ent_vecs", dest='train_ent_vecs', action='store_false')
    parser.set_defaults(train_ent_vecs=False)

    parser.add_argument("--steps_before_evaluation", type=int, default=10000)
    parser.add_argument("--evaluation_minutes", type=int, default=args_dict['evaluation_minutes'], help="every this number of minutes pause"
                                                                           " training and run an evaluation epoch")
    parser.add_argument("--dim_char", type=int, default=args_dict['dim_char'])
    parser.add_argument("--hidden_size_char", type=int, default=args_dict['hidden_size_char'], help="lstm on chars")
    parser.add_argument("--hidden_size_lstm", type=int, default=args_dict['hidden_size_lstm'], help="lstm on word embeddings")

    parser.add_argument("--use_chars", dest="use_chars", action='store_true', help="use character embeddings or not")
    parser.add_argument("--no_use_chars", dest="use_chars", action='store_false')
    parser.set_defaults(use_chars=True)
    parser.add_argument("--model_heads_from_bilstm", type=bool, default=False,
                        help="use the bilstm vectors for the head instead of the word embeddings")
    parser.add_argument("--span_boundaries_from_wordemb", type=bool, default=False, help="instead of using the "
                                                                                         "output of contextual bilstm for start and end of span we use word+char emb")
    parser.add_argument("--span_emb", default=args_dict['span_emb'], help="boundaries for start and end, and head")

    parser.add_argument("--max_mention_width", type=int, default=10)
    parser.add_argument("--use_features", type=bool, default=False, help="like mention width")
    parser.add_argument("--feature_size", type=int, default=20)  # each width is represented by a vector of that size

    parser.add_argument("--ent_vecs_regularization", default=args_dict['ent_vecs_regularization'], help="'no', "
                                                                               "'dropout', 'l2', 'l2dropout'")

    parser.add_argument("--span_emb_ffnn", default="0_0", help="int_int  the first int"
                                                               "indicates the number of hidden layers and the second the hidden size"
                                                               "so 2_100 means 2 hidden layers of width 100 and then projection to output size"
                                                               ". 0_0 means just projecting without hidden layers")
    parser.add_argument("--final_score_ffnn", default=args_dict['final_score_ffnn'], help="int_int  look span_emb_ffnn")

    parser.add_argument("--gamma_thr", type=float, default=0.2)

    parser.add_argument("--nocheckpoints", type=bool, default=False)
    parser.add_argument("--checkpoints_num", type=int, default=1, help="maximum number of checkpoints to keep")

    parser.add_argument("--ed_datasets", default="")
    parser.add_argument("--ed_val_datasets", default="1", help="based on these datasets pick the optimal"
                                                               "gamma thr and also consider early stopping")
    # --ed_val_datasets=1_4  # aida_dev, aquaint
    parser.add_argument("--el_datasets", default=args_dict['el_datasets'])
    parser.add_argument("--el_val_datasets", default=args_dict['el_val_datasets'])  # --el_val_datasets=1_4   # aida_dev, aquaint

    parser.add_argument("--train_datasets", default=args_dict['train_datasets'])
    # --train_datasets=aida_train.txt_z_wikidumpRLTD.txt

    parser.add_argument("--continue_training", type=bool, default=False,
                        help="if true then just restore the previous command line"
                             "arguments and continue the training in exactly the"
                             "same way. so only the experiment_name and "
                             "training_name are used from here. Retrieve values from"
                             "latest checkpoint.")
    parser.add_argument("--onleohnard", type=bool, default=False)

    parser.add_argument("--comment", default="", help="put any comment here that describes your experiment"
                                                      ", for logging purposes only.")

    parser.add_argument("--all_spans_training", type=bool, default=args_dict['all_spans_training'])
    parser.add_argument("--fast_evaluation", type=bool, default=args_dict['fast_evaluation'], help="if all_spans training then evaluate only"
                                                                            "on el tests, corresponding if gm training evaluate only on ed tests.")

    parser.add_argument("--entity_extension", default=None, help="extension_entities or extension_entities_all etc")

    parser.add_argument("--nn_components", default=args_dict['nn_components'], help="each option is one scalar, then these are fed to"
                                                                    "the final ffnn and we have the final score. choose any combination you want: e.g"
                                                                    "pem_lstm_attention_global, pem_attention, lstm_attention, pem_lstm_global, etc")
    parser.add_argument("--attention_K", type=int, default=args_dict['attention_K'], help="K from left and K from right, in total 2K")
    parser.add_argument("--attention_R", type=int, default=args_dict['attention_R'], help="hard attention")
    parser.add_argument("--attention_use_AB", type=bool, default=False)
    parser.add_argument("--attention_on_lstm", type=bool, default=False, help="instead of using attention on"
                                                                              "original pretrained word embedding. use it on vectors or lstm, "
                                                                              "needs also projection now the context vector x_c to 300 dimensions")
    parser.add_argument("--attention_ent_vecs_no_regularization", type=bool, default=args_dict['attention_ent_vecs_no_regularization'])
    parser.add_argument("--attention_retricted_num_of_entities", type=int, default=None,
                        help="instead of using 30 entities for creating the context vector we use only"
                             "the top x number of entities for reducing noise.")
    parser.add_argument("--global_thr", type=float, default=args_dict['global_thr'])  # 0.0, 0.05, -0.05, 0.2
    parser.add_argument("--global_mask_scale_each_mention_voters_to_one", type=bool, default=False)
    parser.add_argument("--global_topk", type=int, default=None)
    parser.add_argument("--global_gmask_based_on_localscore", type=bool, default=False)  # new
    parser.add_argument("--global_topkthr", type=float, default=None)  # 0.0, 0.05, -0.05, 0.2
    parser.add_argument("--global_score_ffnn", default=args_dict['global_score_ffnn'], help="int_int  look span_emb_ffnn")
    parser.add_argument("--global_one_loss", type=bool, default=False)
    parser.add_argument("--global_norm_or_mean", default="norm")
    parser.add_argument("--global_topkfromallspans", type=int, default=None)
    parser.add_argument("--global_topkfromallspans_onlypositive", type=bool, default=False)
    parser.add_argument("--global_gmask_unambigious", type=bool, default=False)

    parser.add_argument("--hardcoded_thr", type=float, default=None, help="if this is specified then we don't calculate"
                                                                          "optimal threshold based on the dev dataset but use this one.")
    parser.add_argument("--ffnn_dropout", dest="ffnn_dropout", action='store_true')
    parser.add_argument("--no_ffnn_dropout", dest="ffnn_dropout", action='store_false')
    parser.set_defaults(ffnn_dropout=True)
    parser.add_argument("--ffnn_l2maxnorm", type=float, default=None, help="if positive"
                                                                           " then bound the Frobenius norm <= value for the weight tensor of the "
                                                                           "hidden layers and the output layer of the FFNNs")
    parser.add_argument("--ffnn_l2maxnorm_onlyhiddenlayers", type=bool, default=False)

    parser.add_argument("--cand_ent_num_restriction", type=int, default=None, help="for reducing memory usage and"
                                                                                   "avoiding OOM errors in big NN I can reduce the number of candidate ent for each span")
    # --ed_datasets=  --el_datasets="aida_train.txt_z_aida_dev.txt"     which means i can leave something empty
    # and i can also put "" in the cla

    parser.add_argument("--no_p_e_m_usage", type=bool, default=False, help="use similarity score instead of "
                                                                           "final score for prediction")
    parser.add_argument("--pem_without_log", type=bool, default=False)
    parser.add_argument("--pem_buckets_boundaries", default=None,
                        help="example: 0.03_0.1_0.2_0.3_0.4_0.5_0.6_0.7_0.8_0.9_0.99")
    # the following two command line arguments
    parser.add_argument("--gpem_without_log", type=bool, default=False)
    parser.add_argument("--gpem_buckets_boundaries", default=None,
                        help="example: 0.03_0.1_0.2_0.3_0.4_0.5_0.6_0.7_0.8_0.9_0.99")
    parser.add_argument("--stage2_nn_components", default="local_global",
                        help="each option is one scalar, then these are fed to"
                             "the final ffnn and we have the final score. choose any combination you want: e.g"
                             "pem_local_global, pem_global, local_global, global, etc")
    parser.add_argument("--ablations", type=bool, default=False)
    args = parser.parse_args()

    if args.training_name is None:
        from datetime import datetime
        args.training_name = "{:%d_%m_%Y____%H_%M}".format(datetime.now())

    temp = "all_spans_" if args.all_spans_training else ""
    args.output_folder = config.base_folder + "data/tfrecords/" + \
                         args.experiment_name + "/{}training_folder/".format(temp) + \
                         args.training_name + "/"

    if args.continue_training:
        print("continue training...")
        train_args = load_train_args(args.output_folder, "train_continue")
        return train_args
    args.running_mode = "train"  # "evaluate"  "ensemble_eval"  "gerbil"

    if os.path.exists(args.output_folder) and not args.continue_training:
        print("!!!!!!!!!!!!!!\n"
              "experiment: ", args.output_folder, "already exists and args.continue_training=False."
                                                  "folder will be deleted in 20 seconds. Press CTRL+C to prevent it.")
        time.sleep(20)
        import shutil
        shutil.rmtree(args.output_folder)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    args.checkpoints_folder = args.output_folder + "checkpoints/"
    if args.onleohnard:
        args.checkpoints_folder = "/cluster/home/nkolitsa/checkpoints_folder/" + \
                                  args.experiment_name + "/" + args.training_name + "/"

    args.summaries_folder = args.output_folder + "summaries/"
    if not os.path.exists(args.summaries_folder):
        os.makedirs(args.summaries_folder)

    args.ed_datasets = args.ed_datasets.split('_z_') if args.ed_datasets != "" else None
    args.el_datasets = args.el_datasets.split('_z_') if args.el_datasets != "" else None
    args.train_datasets = args.train_datasets.split('_z_') if args.train_datasets != "" else None

    args.ed_val_datasets = [int(x) for x in args.ed_val_datasets.split('_')]
    args.el_val_datasets = [int(x) for x in args.el_val_datasets.split('_')]

    args.span_emb_ffnn = [int(x) for x in args.span_emb_ffnn.split('_')]
    args.final_score_ffnn = [int(x) for x in args.final_score_ffnn.split('_')]
    args.global_score_ffnn = [int(x) for x in args.global_score_ffnn.split('_')]

    args.eval_cnt = 0
    args.zero = 1e-6

    if args.pem_buckets_boundaries:
        args.pem_buckets_boundaries = [float(x) for x in args.pem_buckets_boundaries.split('_')]
    if args.gpem_buckets_boundaries:
        args.gpem_buckets_boundaries = [float(x) for x in args.gpem_buckets_boundaries.split('_')]

    if args.fast_evaluation:
        if args.all_spans_training:  # destined for el so omit the evaluation on ed
            args.ed_datasets = None
        else:
            args.el_datasets = None
    return args



def preprocessing(dataset_name, input_folder, output_folder,
                  tokenizer_path="../data/stanford_core_nlp/stanford-corenlp-full-2017-06-09"):
    # Need to pre-process the datasets before reading them
    # Preprocessing code is different for AIDA and for all others


    if dataset_name == "AIDA":
        args = _parse_args_aida(input_folder, output_folder)
        create_necessary_folders()
        # Path of the directory of AIDA data
        # Path of output file (pre processed file)
        process_aida(args.aida_folder + "aida_train.txt", args.output_folder + "aida_train.txt")
        split_dev_test(args.aida_folder+"testa_testb_aggregate_original")

        process_aida(args.output_folder + "temp_aida_dev", args.output_folder + "aida_dev.txt")
        process_aida(args.output_folder + "temp_aida_test", args.output_folder + "aida_test.txt")

        os.remove(args.output_folder + "temp_aida_dev")
        os.remove(args.output_folder + "temp_aida_test")

    else:
        args = _parse_args_others(input_folder, output_folder, tokenizer_path)
        create_necessary_folders()
        processDataset = ProcessDataset()
        for dataset in get_immediate_subdirectories(input_folder):
            processDataset.process(os.path.join(args.other_datasets_folder, dataset))
            print("Dataset ", dataset, "done.")


def main(input_folder):
    end_to_end_el_obj = End_to_end_neural_el()

    # dataset could either be AIDA, ACE2004, AQUAINT, MSNBC or OTHER
    dataset_name = "AIDA"
    # input_folder = "../data/basic_data/test_datasets/AIDA/"
    output_folder = "../data/new_datasets/"
    preprocessing(dataset_name, input_folder, output_folder)
    prepro_util_main()

    # Read the preprocessed data
    # Prepare Arguments for training - These were  read from command line in the original implementation
    args_dict = {}
    args_dict['batch_size'] = 4
    args_dict['experiment_name'] = "paper_models"
    args_dict['training_name'] = "group_global / global_model_v$v"
    args_dict['ent_vecs_regularization'] = "l2dropout"
    args_dict['evaluation_minutes'] = 10
    args_dict['nepoch_no_imprv'] = 6
    args_dict['span_emb'] = "boundaries"
    args_dict['dim_char'] = 50
    args_dict['hidden_size_char'] = 50
    args_dict['hidden_size_lstm'] = 150
    args_dict['nn_components'] = "pem_lstm_attention_global"
    args_dict['fast_evaluation'] = True
    args_dict['all_spans_training'] = True
    args_dict['attention_ent_vecs_no_regularization'] = True
    args_dict['final_score_ffnn'] = "0_0"
    args_dict['attention_R'] = 10
    args_dict['attention_K'] = 100
    args_dict['train_datasets'] = "aida_train"
    args_dict['el_datasets'] = "aida_dev_z_aida_test_z_aida_train"
    args_dict['el_val_datasets'] = 0
    args_dict['global_thr'] = 0.001
    args_dict['global_score_ffnn'] = "0_0"

    # Training for EL task
    args = _parse_args_train(args_dict)

    folder = "../data/tfrecords/"+args_dict['experiment_name']+\
                   ("/allspans/" if args.all_spans_training else "/gmonly/")
    dataset_path = [folder + file for file in args.train_datasets]
    training_dataset = end_to_end_el_obj.read_dataset(dataset_path, args_dict)

    trained_model_el = end_to_end_el_obj.train([training_dataset])

    # # Training for ED task
    args_dict['all_spans_training'] = False
    args = _parse_args_train(args_dict)

    folder = "../data/tfrecords/" + args_dict['experiment_name'] + \
             ("/allspans/" if args.all_spans_training else "/gmonly/")
    dataset_path = [folder + file for file in args.train_datasets]
    training_dataset = end_to_end_el_obj.read_dataset(dataset_path, args_dict)
    trained_model_ed = end_to_end_el_obj.train([training_dataset])


    # Predictions
    prediction_datasets = ["aida_train.txt_z_aida_dev.txt_z_aida_test.txt"]  # <-- as we need list input
    predictions = end_to_end_el_obj.predict(trained_model_el, prediction_datasets)

    # For evaluation
    base_folder = "../../data/tfrecords/"
    dev_set = "aida_dev.txt"
    test_set = "aida_test.txt"
    required_data = [base_folder, dev_set, test_set]
    precision, recall, F1 = end_to_end_el_obj.evaluate(trained_model_el, required_data)

    print ("Accuracy :", precision, recall, F1)

if __name__ == '__main__':
    input_file_path = "../data/basic_data/test_datasets/AIDA/"
    main(input_file_path)