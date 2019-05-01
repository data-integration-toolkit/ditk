import fileinput
import re

from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from gensim.models import KeyedVectors
from breds.seed import Seed
from breds.reverb import Reverb

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"


class Config(object):

    def __init__(self, config_file, positive_seeds, negative_seeds,
                 similarity, confidence):

        # http://www.ling.upenn.edu/courses/Fall_2007/ling001/penn_treebank_pos.html
        # select everything except stopwords, ADJ and ADV
        self.filter_pos = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB']
        self.regex_clean_simple = re.compile('</?[A-Z]+>', re.U)
        self.regex_clean_linked = re.compile('</[A-Z]+>|<[A-Z]+ url=[^>]+>', re.U)
        self.tags_regex = re.compile('</?[A-Z]+>', re.U)
        self.positive_seed_tuples = set()
        self.negative_seed_tuples = set()
        self.vec_dim = 0
        self.e1_type = None
        self.e2_type = None
        self.stopwords = stopwords.words('english')
        self.lmtzr = WordNetLemmatizer()
        self.threshold_similarity = similarity
        self.instance_confidence = confidence
        self.reverb = Reverb()
        self.word2vec = None
        self.vec_dim = None

        # simple tags, e.g.:
        # <PER>Bill Gates</PER>
        self.regex_simple = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)

        # linked tags e.g.:
        # <PER url=http://en.wikipedia.org/wiki/Mark_Zuckerberg>Zuckerberg</PER>
        self.regex_linked = re.compile('<[A-Z]+ url=[^>]+>[^<]+</[A-Z]+>', re.U)

        self.wUpdt = config_file["wUpdt"]
        self.wUnk = config_file["wUnk"]

        self.wNeg = config_file['wNeg']

        self.number_iterations = config_file['number_iterations']

        self.min_pattern_support = config_file['min_pattern_support']

        self.max_tokens_away = config_file['max_tokens_away']

        self.min_tokens_away = config_file['min_tokens_away']

        self.context_window_size = config_file['context_window_size']

        self.word2vecmodelpath = config_file['word2vec_path']

        self.alpha = config_file['alpha']

        self.beta = config_file['beta']
        self.gamma = config_file['gamma']

        self.tag_type = config_file['tag_type']

        assert self.alpha+self.beta+self.gamma == 1

        self.read_seeds(positive_seeds, self.positive_seed_tuples)
        self.read_seeds(negative_seeds, self.negative_seed_tuples)
        fileinput.close()

        print("Configuration parameters")
        print("========================\n")

        print("Relationship/Sentence Representation")
        print("e1 type              :", self.e1_type)
        print("e2 type              :", self.e2_type)
        print("tags type            :", self.tag_type)
        print("context window       :", self.context_window_size)
        print("max tokens away      :", self.max_tokens_away)
        print("min tokens away      :", self.min_tokens_away)
        print("Word2Vec Model       :", self.word2vecmodelpath)

        print("\nContext Weighting")
        print("alpha                :", self.alpha)
        print("beta                 :", self.beta)
        print("gamma                :", self.gamma)

        print("\nSeeds")
        print("positive seeds       :", len(self.positive_seed_tuples))
        print("negative seeds       :", len(self.negative_seed_tuples))
        print("negative seeds wNeg  :", self.wNeg)
        print("unknown seeds wUnk   :", self.wUnk)

        print("\nParameters and Thresholds")
        print("threshold_similarity :", self.threshold_similarity)
        print("instance confidence  :", self.instance_confidence)
        print("min_pattern_support  :", self.min_pattern_support)
        print("iterations           :", self.number_iterations)
        print("iteration wUpdt      :", self.wUpdt)
        print("\n")

    def read_word2vec(self):
        print("Loading word2vec model ...\n")
        self.word2vec = KeyedVectors.load_word2vec_format(self.word2vecmodelpath, binary=True)
        self.vec_dim = self.word2vec.vector_size
        print(self.vec_dim, "dimensions")

    def read_seeds(self, seeds_file, holder):
        for line in fileinput.input(seeds_file):
            if line.startswith("#") or len(line) == 1:
                continue
            if line.startswith("e1"):
                self.e1_type = line.split(":")[1].strip()
            elif line.startswith("e2"):
                self.e2_type = line.split(":")[1].strip()
            else:
                e1 = line.split(";")[0].strip()
                e2 = line.split(";")[1].strip()
                seed = Seed(e1, e2)
                holder.add(seed)