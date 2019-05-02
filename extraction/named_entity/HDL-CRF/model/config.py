import os
from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
        get_processing_word

class Config():
    def __init__(self, input_file_path='', load=True):
        if input_file_path == '':
            self.filename_dev = "data/dev_set.iob"
            self.filename_test = "data/test_set.iob"
            self.filename_train = "data/train_set.iob"
        else:
            self.filename_dev = input_file_path
            self.filename_test = input_file_path
            self.filename_train = input_file_path

        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()


    def load(self):
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)


    # general config
    dir_output = "results/test/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"

    # embeddings
    dim_word = 200
    dim_char = 25

    # glove files
    filename_glove = "data/glove.6B.{}d.txt".format(dim_word)
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = "data/word2vec.40B.{}d.trimmed.npz".format(dim_word)
    use_pretrained = True

    # dataset
    '''
    filename_dev = "data/dev_set.iob"
    filename_test = "data/test_set.iob"
    filename_train = "data/train_set.iob"
    '''

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "data/words.txt"
    filename_tags = "data/tags.txt"
    filename_chars = "data/chars.txt"

    # training
    train_embeddings = False
    nepochs          = 1
    dropout          = 0.5
    batch_size       = 20
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             =  -1
    nepoch_no_imprv  = 3

    # model hyperparameters
    hidden_size_char = 25
    hidden_size_lstm = 100


    use_crf = True
    use_char_lstm = False
    use_char_cnn=True
   
    filter_size=[2,3]
    filter_deep=30
