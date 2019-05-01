from entity_linkage.normalization.entity_normalization import EntityNormalization
from entity_linkage.normalization.lnex import read_data
from entity_linkage.normalization.lnex import eval_main


import core as lnex
import os,sys

class LNEx(EntityNormalization):

    @classmethod
    def read_dataset(cls, dataset_name, split_ratio=(1,0,0), options={}):
        '''
        File name provided to the function to read the dataset. Here dataset could be a stream of Tweet or benchmark
        dataset
        
        :param dataset_name: name of dataset - a string
        :param split_ratio: (train_ratio, validation_ration, test_ratio)
        :optional arguemts for this class specified using *args
        :isTwitter : Boolean to check if data read is from twitter or other benchmark dataset
        :eventLocation : Provide this information for blocking
        :return: train_data, valid_data, test_data
        '''
        
        train_data, test_data = read_data.read_data(dataset_name)
        return train_data, test_data
    
        


    @classmethod
    def train(cls, train_dev_set):
        '''
        This method will prepare gazetteer as it as a statistical model no training is required
        
        :param train_set: set to null 
        :return: t gazetteer initialized
        '''   
        bbs = { "chennai": [12.74, 80.066986084, 13.2823848224, 80.3464508057]}

        dataset = "chennai"
        #lnex.elasticindex(conn_string='localhost:9200', index_name="photon")
        geo_info = lnex.initialize( bbs[dataset], augmentType="HP",
                                    cache=False,
                                    dataset_name=dataset,
                                    capital_word_shape=False)
        return geo_info


    @classmethod
    def predict(cls, model,test_set):
        '''
        The method extracts location based on the statistical model and outputs the following results
        
        :param model: a trained model
        :param test_set: a list of test data
        :return: returns a list of the following 4 items list:
            tweet_mention, mention_offsets, geo_location, geo_info_id
            tweet_mention:   is the location mention in the tweet
                             (substring retrieved from the mention offsets)
            mention_offsets: a tuple of the start and end offsets of the LN
            geo_location:    the matched location name from the gazetteer.
                             e.g., new avadi rd > New Avadi Road
            geo_info_id:  s   contains the attached metadata of all the matched
                             location names from the gazetteer
        '''
        
        for tweet in test_set:
            for output in lnex.extract(tweet):
                print(output[0], output[1], output[2], output[3]["main"])
            print("#"*50)
                  
        ditk_path = ""
        for path in sys.path:
            if "ditk" in path:
                ditk_path = path
                
        output_file = ditk_path+"/entity_linkage/normalization/sgtb/result/output.txt"
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, "w") as f:
            for tweet in test_set:
                for output in lnex.extract(tweet):
                    f.write(output[0] + ", " + output[1] + ", " + output[2] + ", " + output[3]["main"] + "\n")
        
        return output
        
    
    @classmethod
    def evaluate(cls, clf, eval_set):
        '''
        :param model: a trained model
        :param eval_set: a list of validation data
        :return: (precision, recall, f1 score)
        '''       
        bbs = { "chennai": [12.74, 80.066986084, 13.2823848224, 80.3464508057]}

  
        results = dict()
        dataset = "chennai"
        
        print(dataset)
        lnex.initialize( bbs[dataset], augmentType="FULL",
                            cache=False,
                            dataset_name=dataset,
                            capital_word_shape=False)
        
                
        anns = eval_set
        
        results = eval_main.evaluate(anns)
        
            
        return results


    @classmethod
    def save_model(cls, clf, file_name):
        pass

    @classmethod
    def load_model(cls, file_name):
        pass
    
