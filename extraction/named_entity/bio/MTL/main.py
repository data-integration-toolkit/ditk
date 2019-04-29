from __future__ import print_function

import random
from logging import info
import sys

from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, merge
from keras.layers import Reshape, Convolution2D, Dropout

from models.ltlib import filelog
from models.ltlib import conlldata
from models.ltlib import viterbi

from models.ltlib.features import NormEmbeddingFeature, SennaCapsFeature
from models.ltlib.features import windowed_inputs
from models.ltlib.callbacks import token_evaluator, EpochTimer
from models.ltlib.layers import concat, inputs_and_embeddings
from models.ltlib.settings import cli_settings, log_settings
from models.ltlib.optimizers import get_optimizer
from models.ltlib.output import save_token_predictions

from models.config import Defaults
from sklearn.metrics import precision_recall_fscore_support, f1_score

sys.argv = ["single_task.py", "./data/CHEMDNER", "./vectorfile/bio_nlp_vec/PubMed-shuffle-win-30.bin"]
reload(sys)
sys.setdefaultencoding('utf-8')

class MTL:
    def __init__(self):
        self.config = cli_settings(['datadir', 'wordvecs'], Defaults)
        
    def convert_ground_truth(self, data, *args, **kwargs):  # <--- implemented PER class
        pass

    def read_dataset(self, file_dict, dataset_name):  # <--- implemented PER class
        data = conlldata.load_dir(self.config.datadir, self.config)
        return data
            
            
    def train(self, data):  # <--- implemented PER class
        if self.config.viterbi:
            vmapper = viterbi.TokenPredictionMapper(data.train.sentences)
        else:
            vmapper = None

        w2v = NormEmbeddingFeature.from_file(self.config.wordvecs,
                                             max_rank=self.config.max_vocab_size,
                                             vocabulary=data.vocabulary,
                                             name='words')
        features = [w2v]
        if self.config.word_features:
            features.append(SennaCapsFeature(name='caps'))

        data.tokens.add_features(features)
        data.tokens.add_inputs(windowed_inputs(self.config.window_size, features))

        # Log word vector feature stat summary
        #info('{}: {}'.format(self.config.wordvecs, w2v.summary()))

        inputs, embeddings = inputs_and_embeddings(features, self.config)

        seq = concat(embeddings)
        cshape = (self.config.window_size, sum(f.output_dim for f in features))

        seq_reshape = Reshape(cshape+(1,))(seq)


        # Convolutions
        conv_outputs = []
        for filter_size, filter_num in zip(self.config.filter_sizes, self.config.filter_nums):
            conv = Convolution2D(filter_num, filter_size, cshape[1],
                                 activation='relu')(seq_reshape)
            cout = Flatten()(conv)
            conv_outputs.append(cout)
        seq_covout = merge(conv_outputs, mode='concat', concat_axis=-1)


        for size in self.config.hidden_sizes:
            seq_2 = Dense(size, activation=self.config.hidden_activation)(seq_covout)
        seq_3 = Dropout(self.config.output_drop_prob)(seq_2)
        out = Dense(data.tokens.target_dim, activation='softmax')(seq_3)
        self.model = Model(input=inputs, output=out)


        optimizer = get_optimizer(self.config)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])

        callbacks = [
            EpochTimer(),
            #token_evaluator(data.train, mapper=vmapper, config=config),
            #token_evaluator(data.test, mapper=vmapper, config=config),
        ]

        percnt_keep = self.config.percent_keep
        amt_keep = len(data.train.tokens.inputs['words']) * percnt_keep
        print("Total: %s. Keeping: %s" % (len(data.train.tokens.inputs['words']), amt_keep))
        start = random.randrange(int(len(data.train.tokens.inputs['words']) - amt_keep + 1))
        end = int(start + amt_keep)
        x = data.train.tokens.inputs['words'][start:end]


        self.model.fit(
            #data.train.tokens.inputs,
            x,
            data.train.tokens.targets[start:end],
            callbacks=callbacks,
            batch_size=self.config.batch_size,
            nb_epoch=1,
            verbose=1
        )
      
    def predict(self, data, *args, **kwargs):  # <--- implemented PER class WITH requirement on OUTPUT format!
        name = "chemdner_output"
        save_token_predictions(name, data.test, self.model, conlldata.write)

        return "./prediction"+name+".tsv"

    def evaluate(self, predictions, groundTruths, *args,
                 **kwargs):  # <--- common ACROSS ALL classes. Requirement that INPUT format uses output from predict()!
        f = open("./prediction/chemdner_output.tsv", "r")
        lines = f.readlines()
        f.close()

        ground = []
        predict = []
        for line in lines:
            if len(line) < 2:
                 continue
            tmp = line.split(" ")
            ground.append(tmp[1])
            predict.append(tmp[2].strip())
        
        eval = precision_recall_fscore_support(ground, predict, average='macro', labels=list(set(predict)))
        test_score = eval[2]
        
        return test_score


    def save_model(self, filepath):
        pass
    

    def load_model(self, filepath):
        pass

def main(input_path):
        
    MTL_instance = MTL()

    read_data = MTL_instance.read_dataset(None, None)

    MTL_instance.train(read_data)


    output_file = MTL_instance.predict(read_data)
    print("Output file has been created at: {}".format(output_file))

    f1_score = MTL_instance.evaluate(None, None)
    print("f1: {}".format(f1_score))

    return output_file

if __name__ == '__main__':
    main("./data/CHEMDNER")



