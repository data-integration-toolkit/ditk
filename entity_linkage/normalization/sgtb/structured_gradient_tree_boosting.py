from entity_normalization import EntityNormalization
import read_data
from structured_gradient_boosting import StructuredGradientBoosting
import gzip, time, os
from sklearn.externals import joblib


class StructuredGradientTreeBoosting(EntityNormalization):

    @classmethod
    def read_dataset(cls, dataset_name, split_ratio, options={}):
        '''
        :param dataset_name (str): name of dataset
        :param split_ratio (tuple): (train_ratio, validation_ration, test_ratio)
        :param kwargs: other parameters for specific model (optional)
        :return (tuple): train_data, valid_data, test_data
        '''
        train_data, valid_data, test_data = read_data.read_data(dataset_name, split_ratio, options)
        return train_data, valid_data, test_data


    @classmethod
    def train(cls, train_dev_set):
        '''
        :param train_set (list): a list of training data
        :return (Model): trained model
        '''     

        print("Loading features...")

        train_set, dev_set = train_dev_set

        # entity-entity features
        ent_ent_feat_dict = {}
        with gzip.open("./data/ent_ent_feats.txt.gz", 'rb') as f:
            for line in f:
                ep, feat_str = line.split(b'\t')
                e1, e2 = ep.split()
                feats = [float(x) for x in feat_str.split()]
                ent_ent_feat_dict[(e1,e2)] = feats

        print("Loading features... Finished!")
        print("Start training...")

        clf = StructuredGradientBoosting(max_depth=3,
                                         learning_rate=1.0,
                                         n_estimators=250,
                                         min_samples_split=2,
                                         min_samples_leaf=1,
                                         ent_ent_feat_dict=ent_ent_feat_dict,
                                         beam_width=4,
                                         num_thread=8)
        
        start_time = time.time()
        clf = clf.fit(train_set, dev_set)
        end_time = time.time()
        print ("Training take %.2f secs" %(end_time - start_time))   
        return clf


    @classmethod
    def predict(cls, clf, test_set):
        '''
        :param model (Model): a trained model
        :param test_set (list): a list of test data
        :return (list): a list of prediction, each item with the format
        (entity_name, wikipedia_url(optional), geolocation_url(optional), geolocation_boundary(optional))
        '''
        test_X, _, test_indices, test_ent_ids = test_set
        test_pred = clf.predict(test_X, test_indices, test_ent_ids)

        output = []
        for pred in test_pred:
            output.append((pred, "", "", ""))

        output_file = "./data/output.txt"
        with open(output_file, "w") as f:
            for entity, wiki_url, geo_url, geo_bnd in output:
                f.write(entity + ", " + wiki_url + ", " + geo_url + ", " + geo_bnd + "\n")

        return output
        
    """
    TODO:
    1. implement F1 metrics that originally not available
    """
    @classmethod
    def evaluate(cls, clf, eval_set):
        '''
        :param model (Model): a trained model
        :param eval_set (list): a list of validation data
        :return (tuple): (precision, recall, f1 score)
        '''        
        eval_X, eval_y, eval_indices, eval_ent_ids = eval_set
        eval_acc = clf.get_acc(eval_X, eval_y, eval_indices, eval_ent_ids)
        print ("Test acc %.2f" %(eval_acc))
        return eval_acc


    @classmethod
    def save_model(cls, clf, file_name):
        print("start saving model...")
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
        joblib.dump(clf, file_name)
        print("Finished saving model!")


    def load_model(cls, file_name):
        print("start loading model...")
        clf = joblib.load(file_name)
        print("Finished loading model!")
        
        return clf
