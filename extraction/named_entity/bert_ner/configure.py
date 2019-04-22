import argparse
import sys


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_config_file", default="checkpoint/bert_config.json",
                        type=str, help="The config json file corresponding to the pre-trained BERT model.")
    parser.add_argument("--model_dir", default="output/result_dir/",
                        type=str, help="The output directory where the model checkpoints will be written.")

    # Other parameters
    parser.add_argument("--init_checkpoint", default="checkpoint/bert_model.ckpt",
                        type=str, help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--use_tpu", default=False,
                        type=bool, help="Whether to use TPU or GPU/CPU.")
    parser.add_argument("--max_seq_length", default=128,
                        type=int, help="The maximum total input sequence length after WordPiece tokenization.")
    parser.add_argument("--train_batch_size", default=32,
                        type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8,
                        type=int, help="Total batch size for eval.")
    parser.add_argument("--predict_batch_size", default=8,
                        type=int, help="Total batch size for predict.")
    parser.add_argument("--learning_rate", default=2e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0,
                        type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1,
                        type=float, help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--save_checkpoints_steps", default=1000,
                        type=int, help="How often to save the model checkpoint.")
    parser.add_argument("--iterations_per_loop", default=1000,
                        type=int, help="How many steps to make in each estimator call.")
    parser.add_argument("--vocab_file", default="checkpoint/vocab.txt",
                        type=str, help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--master", default=None,
                        type=str, help="[Optional] TensorFlow master URL.")
    parser.add_argument("--num_tpu_cores", default=8,
                        type=int, help="Only used if `use_tpu` is True. Total number of TPU cores to use.")

    # if len(sys.argv) == 0:
    #     parser.print_help()
    #     sys.exit(1)

    print("")
    args = parser.parse_args()
    for arg in vars(args):
        print("{}={}".format(arg, getattr(args, arg)))
    print("")

    return args

# FLAGS = parse_args()


class ARGUMENTS(object):
    def __init__(self):
        self.bert_config_file = "checkpoint/bert_config.json"
        self.model_dir = "output/result_dir/"
        self.init_checkpoint = "checkpoint/bert_model.ckpt"
        self.use_tpu = False
        self.max_seq_length = 128
        self.train_batch_size = 32
        self.eval_batch_size = 8
        self.predict_batch_size = 8
        self.learning_rate = 2e-05
        self.num_train_epochs = 3.0
        self.warmup_proportion = 0.1
        self.save_checkpoints_steps = 1000
        self.iterations_per_loop = 1000
        self.vocab_file = "checkpoint/vocab.txt"
        self.master = None
        self.num_tpu_cores = 8


FLAGS = ARGUMENTS()
