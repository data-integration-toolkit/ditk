import os


from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
        get_processing_word


class Config():
    def __init__(self, load=True):
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
        #print(self.filename_words)
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=True)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)


    # general config
    dir_output = "source_model/"
    dir_model  = dir_output + "model_weights"
    path_log   = dir_output + "log_source.txt"

    # embeddings
    dim_word = 50
    dim_char = 100

    # glove files
    filename_glove = "resources/source.emb".format(dim_word)
    # trimmed embeddings (created from glove_filename with prep_source_data.py)
    filename_trimmed = "resources/source.emb.trimmed.npz".format(dim_word)
    use_pretrained = False

    # dataset
    filename_train = "datasets/conll2003/train"
    filename_dev = "datasets/conll2003/dev"
    filename_test = "datasets/conll2003/test"


    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with prep_source_data.py)
    filename_words = "src/data/source_words.txt"
    filename_tags = "src/data/source_tags.txt"
    filename_chars = "src/data/source_chars.txt"

    # training
    train_embeddings = True
    nepochs          = 3
    dropout          = 0.5
    batch_size       = 40
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 5 # patient for waiting
    psi = 0.5


    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True # if crf, training is 1.7x slower on CPU
    use_chars = True # if char embedding, training is 3.5x slower on CPU
    use_multigpu = False
    gpu_ids = [0]
