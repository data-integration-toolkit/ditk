from relation_extraction import RelationExtraction
import os
import shutil
import re
import src.train as src_train
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

class CNNModel(RelationExtraction):

    def __init__(self):
        RelationExtraction.__init__(self)
        self.relation_types_dict = {}

    def data_preprocess(self, input_data):
        input_files = [input_data+"/relation_extraction_input_train.txt", input_data+"/relation_extraction_input_test.txt"]
        output_dir = input_data

        dir_path = os.path.dirname(__file__)
        embedding_files = os.listdir(dir_path+"\\embeddings")
        for file_name in embedding_files:
            file_name_full = os.path.join(dir_path+"\\embeddings", file_name)
            if (os.path.isfile(file_name_full)):
                shutil.copy(file_name_full, output_dir)

        data = []
        relation_types_list = []
        for i in range(len(input_files)):
            _data = []
            with open(input_files[i], 'r') as file1:
                lines = file1.readlines()
                for line in lines:
                    attrs = line.strip().lower().split('\t')
                    sentence, e1, _, e1_start_char, _, e2, _, e2_start_char, _, relation_raw = attrs
                    e1_w = e1.split(" ")
                    e1_start_pos = len(sentence[:int(e1_start_char)].strip().split(" "))
                    e1_end_pos = e1_start_pos+len(e1_w)-1
                    e2_w = e2.split(" ")
                    e2_start_pos = len(sentence[:int(e2_start_char)].strip().split(" "))
                    e2_end_pos = e2_start_pos+len(e2_w)-1
                    if relation_raw not in self.relation_types_dict:
                        index = len(self.relation_types_dict)
                        self.relation_types_dict[relation_raw] = index
                        relation_types_list.append(str(index)+" "+relation_raw)              
                    relation = self.relation_types_dict[relation_raw]
                    datum = " ".join([str(relation), str(e1_start_pos), str(e1_end_pos), str(e2_start_pos), str(e2_end_pos), sentence])
                    _data.append(datum)
                
                data.append(_data)

                file1.close()

        with open(output_dir+"/relations.txt", 'w') as file2:
            for relation in relation_types_list:
                file2.write(relation)
                file2.write("\n")
                    
            file2.close()
        
        with open(output_dir+"/train.cln", 'w') as file3:
            for datum in data[0]:
                file3.write(datum)
                file3.write("\n")
                    
            file3.close()
        
        with open(output_dir+"/test.cln", 'w') as file4:
            for datum in data[1]:
                file4.write(datum)
                file4.write("\n")
                    
            file4.close()
    
    def read_dataset(self, input_file):
        with open(input_file+"/relations.txt", 'r') as file1:
            for line in file1:
                index, relation_raw  = line.strip().split(" ")
                self.relation_types_dict[relation_raw] = int(index)
        
            file1.close()

    def tokenize(self, input_data, ngram_size=None):
        pass

    def train(self, train_data, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        sample_size = 0
        with open(train_data+"/train.cln", 'r') as file_train:
            for _ in file_train:
                sample_size += 1
            file_train.close()

        params = {
            'data': train_data, 
            'sample_size': sample_size,
            'num_relations': len(self.relation_types_dict),
            'save_model_folder': model_dir, 
            'test': False
        }

        src_train.train(params)

        return model_dir
    
    def save_model(self, file):
        '''
            Deprecated as model is automatically saved to disk in function train()
        '''
        pass
    
    def load_model(self, file):
        '''
            Deprecated as model is automatically loaded from disk in function predit()
        '''
        pass

    def predict(self, test_data, entity_1=None, entity_2=None,  trained_model=None):
        output_dir = test_data
        e1s = []
        e2s = []
        sentences = []
        relations_real = []
        relations_predict = []

        sample_size = 0
        with open(test_data+"/relation_extraction_input_test.txt", 'r') as file_test:
            for line in file_test:
                attrs = line.strip().lower().split('\t')
                sentence, e1, _, _, _, e2, _, _, _, relation_real = attrs
                e1s.append(e1)
                e2s.append(e2)
                sentences.append(sentence)
                relations_real.append(relation_real)
                sample_size += 1
            file_test.close()

        params = {
            'data': test_data,
            'sample_size': sample_size,
            'num_relations': len(self.relation_types_dict),
            'save_model_folder': trained_model, 
            'test': True
        }

        src_train.train(params)

        with open(output_dir+"/results.txt", 'r') as results_raw:
            for line in results_raw:
                _, relation_predict = line.strip().split("\t")
                relations_predict.append(relation_predict)
            results_raw.close()
        
        output_file_path = output_dir+"/relation_extraction_output.txt"
        with open(output_file_path, 'w') as output_file:
            for iter in zip(sentences, e1s, e2s, relations_predict, relations_real):
                output_file.write("\t".join(iter))
                output_file.write("\n")
                    
            output_file.close()

        return output_file_path

    def evaluate(self, input_data, trained_model=None):
        dir_output = input_data
        y_pred = []
        y_true = []
        
        with open(dir_output+"/results.txt", 'r') as file3:
            for line in file3:
                _, relation_raw = line.strip().split("\t")
                y_pred.append(self.relation_types_dict[relation_raw])
            
            file3.close()
        
        with open(dir_output+"/test.cln", 'r') as file4:
            for line in file4:
                relation_id = line.split(" ")[0]
                y_true.append(int(relation_id))

            file4.close()

        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        
        return (precision, recall, f1)