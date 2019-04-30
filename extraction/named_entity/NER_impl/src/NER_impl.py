import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from named_entity.ner import Ner
import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor
from named_entity.NER_impl.src.All_other_methods import *
from named_entity.NER_impl.src.conlleval import *


# PLEASE NOTE: buildGlove() method in All_other_methods requires glove.840b.300d.txt
# which is downloaded from https://nlp.stanford.edu/projects/glove/

class BiLSTMCRFNerImpl(Ner):
    Path('results').mkdir(exist_ok=True)

    def read_dataset(self,file_dict, dataset_name=None, *args, **kwargs):
        """

        :param file_dict: dictionary
                    {
                        "train": "location_of_train",
                        "test": "location_of_test",
                        "dev": "location_of_dev",
                    }

        :param args:
        :param kwargs:
        :return: dictionary of iterables
                    Format:
                    {
                        "train":[
                                    [ Line 1 tokenized],
                                    [Line 2 tokenized],
                                    ...
                                    [Line n tokenized]
                                ],
                        "test": same as train,
                        "dev": same as train
                    }

        """
        standard_split = ["train", "test", "dev"]
        data = {}
        try:
            for split in standard_split:
                file = file_dict[split]
                with open(file, mode='r', encoding='utf-8') as f:
                    raw_data = f.read().splitlines()
                data[split] = raw_data
        except KeyError:
            raise ValueError("Invalid file_dict. Standard keys (train, test, dev)")
        except Exception as e:
            print('Something went wrong.', e)
        return data


    def train(self,data):
        """
        Trains the model on the given input data
        :param data: dict containing contents of train,dev and test files
        :return:None. Trained model stored internally to class instance state.
        """
        read_dataset_withParse(self,data)

        params = {
            'dim': 300,
            'dim_chars': 100,
            'dropout': 0.5,
            'num_oov_buckets': 1,
            'epochs': 25,
            'batch_size': 20,
            'buffer': 15000,
            'char_lstm_size': 25,
            'lstm_size': 100,
            'words': 'vocab.words.txt',
            'chars': 'vocab.chars.txt',
            'tags': 'vocab.tags.txt',
            'glove': 'glove.npz'
        }


        with Path('results/params.json').open('w') as f:
            json.dump(params, f, indent=4, sort_keys=True)

        train_inpf = functools.partial(input_fn, "train_words.txt", "train_tags.txt",
                                       params, shuffle_and_repeat=True)
        eval_inpf = functools.partial(input_fn,"dev_words.txt" , "dev_tags.txt")
        #
        cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
        estimator = tf.estimator.Estimator(model_fn, 'results/model', cfg, params)
        Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
        hook = tf.contrib.estimator.stop_if_no_increase_hook(
            estimator, 'f1', 500, min_steps=8000, run_every_secs=120)
        train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        export_model()

    def predict(self,test_data=None,*args):
        """
        predicts on the given input data based on the trained model
        :param None, loads the saved model
        :return: predictions of test file: [tuple,...], i.e. list of tuples.
                Each tuple is (start index, span, mention text, mention type)
                Where:
                 - start index: int, the index of the first character of the mention span. None if not applicable.
                 - span: int, the length of the mention. None if not applicable.
                 - mention text: str, the actual text that was identified as a named entity. Required.
                 - mention type: str, the entity/mention type. None if not applicable.
        """

        start_index=None
        span=None
        export_dir = 'saved_model'
        # line="Sam lives in New York"
        subdirs = [x for x in Path(export_dir).iterdir()
                   if x.is_dir() and 'temp' not in str(x)]
        latest = str(sorted(subdirs)[-1])
        predict_fn = predictor.from_saved_model(latest)
        final_res=[]
        with open('test_words.txt','r') as fr,open('test_tags.txt','r') as ft,open('ner_test_output.txt','w') as fw:
            lines=fr.readlines()
            tags=ft.readlines()
            for l,t in zip(lines,tags):
                t=t.strip().split(" ")
                predictions=predict_fn(parse_fn_pred(l))
                prediction_tags=predictions["tags"].astype('U13')
                pred_array=prediction_tags.tolist()
                res=[None,None,l,predictions["tags"]]
                for word,ta,pred in zip(l.strip().split(" "),t,pred_array[0]):
                    fw.write(word+" "+ta+" "+pred+"\n")
                final_res.append(res)
                fw.write("\n")
        return final_res


    def convert_ground_truth(self,data=None):
        """
        returns an iterable containing the ground truth tags from the words.txt and tags.txt as required by evaluate
        ie [mention text, corresponding tag]
        :param data: tuple with file contents. Member of data dict obtained after read_dataset
        :return: groundtruth tuple [mention text,mention type]
        """
        final_res=[]
        with open("test_words.txt",'r') as fw,open("test_tags.txt",'r') as ft:
            for w,t in zip(fw.readlines(),ft.readlines()):
                w = w.strip().split(" ")
                t = t.strip().split(" ")
                res=[]
                for word,tag in zip(w,t):
                    res.append(word+" "+tag)
                final_res.append(res)
        fw.close()
        ft.close()
        return final_res

    def evaluate(self,predictions,groundtruth,*args):
        """
        calculates precision,recall and F1 score for the dataset
        :param predictions: output of predict
        :param groundtruth: output of convert_ground_truth
        :return: tuple with [precision,recall,f1]
        """
        if len(predictions) != len(groundtruth):
            print("Length of predictions and ground truth don't match")
        else:
            res=""
            for p,g in zip(predictions,groundtruth):
                p=p[3].tolist()
                sent=""
                for pre,g in zip(p[0],g):
                    sent= sent+g+" "+str(pre)[2:][:-1]+ "\n"
                res = res+ sent+"\n"

        counts=eval(res)
        m=metrics(counts)
        return [m[0][3],m[0][4],m[0][5]]

    def save_model(self, file):
        pass

    def load_model(self, file):
        pass
    
    def main(self,inputfile):
        filedict={"train":inputfile,"dev":inputfile,"test":inputfile}
        obj=BiLSTMCRFNerImpl()
        data=obj.read_dataset(filedict)
        obj.train(data)
        gt=obj.convert_ground_truth(data["test"])
        pred=obj.predict(data)
        metrics=obj.evaluate(pred,gt)
        return "ner_test_output.txt"


