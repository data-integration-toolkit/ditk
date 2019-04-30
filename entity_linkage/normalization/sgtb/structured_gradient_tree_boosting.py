from entity_linkage.normalization.entity_normalization import EntityNormalization
from entity_linkage.normalization.sgtb import read_data
from entity_linkage.normalization.sgtb.structured_gradient_boosting import StructuredGradientBoosting
import gzip, time, os, sys
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
        train_set, dev_set = train_dev_set
        ent_ent_feat_dict = cls._read_feat(cls)

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
        if clf is None:
            raise Exception("model is neither trained nor loaded")

        #find ditk_path from sys.path
        ditk_path = ""
        for path in sys.path:
            if "ditk" in path:
                ditk_path = path

        test_X, _, test_indices, test_ent_ids = test_set
        test_pred = clf.predict(test_X, test_indices, test_ent_ids)

        output = []
        for label, pred in zip(test_indices[0][1], test_pred):
            output.append((label, pred, "", ""))

        output_file = ditk_path+"/entity_linkage/normalization/sgtb/result/output.txt"
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

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
        if clf is None:
            raise Exception("model is neither trained nor loaded")

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
        if not os.path.exists(file_name):
            print("model not exists... init model...")
            ent_ent_feat_dict = cls._read_feat()
            clf = StructuredGradientBoosting(max_depth=3,
                                 learning_rate=1.0,
                                 n_estimators=250,
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 ent_ent_feat_dict=ent_ent_feat_dict,
                                 beam_width=4,
                                 num_thread=8)
            print("Finished initiating model!")
        else:
            print("start loading model...")
            clf = joblib.load(file_name)
            print("Finished loading model!")
            
        return clf

    def _read_feat(cls):
        print("Loading features...")
        #find ditk_path from sys.path
        ditk_path = ""
        for path in sys.path:
            if "ditk" in path:
                ditk_path = path

        # entity-entity features
        feat_file = ditk_path+"/entity_linkage/normalization/sgtb/data/ent_ent_feats.txt.gz"
        ent_ent_feat_dict = {}
        with gzip.open(feat_file, 'rb') as f:
            for line in f:
                ep, feat_str = line.split(b'\t')
                e1, e2 = ep.split()
                feats = [float(x) for x in feat_str.split()]
                ent_ent_feat_dict[(e1,e2)] = feats

        return ent_ent_feat_dict
