# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 22:59:54 2019

@author: konoha
"""
from keras.callbacks import Callback
import prepare_data as prd
import configs
import numpy as np
from seqeval.metrics import f1_score, classification_report
from keras.callbacks import ModelCheckpoint
import model as mdl


class End_to_end():
    def convert_ground_truth (filename, *args, **kwargs):
        data = []
        with open(filename, 'r') as f:  ## ner_test_input.txt
            for line in f.readlines():
                if line == '\n':
                    data.append(' ')
                    continue
                data.append(line.split(' ')[3].rstrip('\n'))
        return data
  
#        with open(filename, 'r+') as f:
#            f.writelines(data)
            
    def read_dataset(filenames, *args, **kwargs):
        ## read the training data and pre-processing
        training_file = filenames[0]
        sentences, output_labels, max_length = prd.read_data(training_file)
        word_to_index = prd.get_vocabulory(sentences)
        label_to_index, index_to_label  = prd.prepare_outputs(output_labels)
        char_index = prd.get_vocabulory(word_to_index)
        max_length = configs.MAX_SEQ_LEN
        char_indices = prd.get_chars(sentences, max_length, char_index)
        vocab_size = len(word_to_index)
        glove_vectors = prd.read_glove_vecs(configs.GLOVE_EMBEDDINGS)
        word_embeddings = prd.get_preTrained_embeddings(word_to_index,glove_vectors,vocab_size)
        max_length = configs.MAX_SEQ_LEN

        
        with open(configs.DICT_FILE, 'w',encoding='utf-8') as file:
            file.write(str(word_to_index))
            file.write("\n")
            file.write(str(label_to_index))
            file.write("\n")
            file.write(str(max_length))
        
        with open(configs.EMBEDDINGS_FILE, 'wb') as file:
            np.save(file, word_embeddings)
        
        #input and output sequences to the model
        train_indeces = prd.get_sequence_indices(sentences, word_to_index, max_length)
        labels  = prd.get_sequence_indices(output_labels, label_to_index, max_length)
        no_of_classes = len(label_to_index)
        no_of_examples = len(sentences)
        print('Total no of input sequences:', no_of_examples)
        assert (len(train_indeces) == len(labels)),"length of I/O sequences doesn't match"
        
        #validation samples/examples
        sentences_v, output_labels_v, max_length_v = prd.read_data(configs.VALIDATION_FILE)
        indeces_v = prd.get_sequence_indices(sentences_v, word_to_index, max_length)
        labels_v  = prd.get_sequence_indices(output_labels_v, label_to_index, max_length)
        char_indices_v = prd.get_chars(sentences_v, max_length, char_index)
        assert (len(indeces_v) == len(labels_v)),"length of I/O sequences doesn't match"
        max_length = configs.MAX_SEQ_LEN
        
        
        return [word_embeddings, char_index, max_length, char_indices, no_of_classes, train_indeces, labels, indeces_v, char_indices_v, labels_v, index_to_label, output_labels_v]
        

    def train(data, *args, **kwargs):
        
        index_to_label = data[10]
        output_labels_v = data[11]
        class Metrics(Callback):
            def on_train_begin(self, logs={}):
                self.val_f1s = []
                self.val_recalls = []
                self.val_precisions = []
     
            def on_epoch_end(self, epoch, logs={}):
                pred_label = np.asarray(self.model.predict(self.validation_data[0:2]))
                pred_label = np.argmax(pred_label,axis=-1)
                #Skipping padded sequences
                pred_label = prd.get_orig_labels(pred_label,index_to_label,output_labels_v)
                result  = f1_score(output_labels_v,pred_label)
                print("F1-score--->",result)
                return
        
        word_embeddings = data[0]
        char_index = data[1]
        max_length = data[2]
        char_indices = data[3]
        no_of_classes = data[4]
        train_indeces = data[5]
        labels = data[6]
        indeces_v = data[7]
        char_indices_v = data[8]
        labels_v = data[9]
        
        
        model = mdl.get_model(word_embeddings, max_length, len(char_index), no_of_classes)
        model.summary()
        
        metrics =  Metrics()
        checkpointer = ModelCheckpoint(configs.MODEL_FILE, monitor = 'val_acc', verbose=1, save_best_only=True,save_weights_only=True, period=3, mode='max')
        
        model.fit(x = [train_indeces,char_indices] , y = np.expand_dims(labels,axis=-1), batch_size=configs.BATCH_SIZE,epochs= configs.EPOCHS,
                  verbose=1, validation_data=([indeces_v,char_indices_v], np.expand_dims(labels_v,axis=-1)), callbacks = [metrics,checkpointer], shuffle=False)
        model.save(configs.MODEL_FILE)
        
        
    def predict(*args, **kwargs):
        def remove_sents(sentences_t,labels_t,max_length):
            remove_idxs = []
            for i,sentence in enumerate(sentences_t):
                if len(sentence)>max_length:
                    remove_idxs.append(i)
            for i in remove_idxs:
                sentences_t.pop(i)
                labels_t.pop(i)
        
        word_index = {}
        label_index = {}
        max_length = 0
        with open(configs.DICT_FILE, 'r', encoding='utf-8') as file:
            dicts  = file.read()
            dicts = dicts.split("\n")
            word_index = eval(dicts[0])
            label_index = eval(dicts[1])
            max_length  = eval(dicts[2])
            
        with open(configs.EMBEDDINGS_FILE, 'rb') as file:
            word_embeddings = np.load(file)
            
        #Loading test sequences    
        sentences_t, labels_t, max_length_t = prd.read_data(configs.TEST_FILE)

        remove_sents(sentences_t, labels_t, max_length)
        print('Total no of test sequences: ', len(sentences_t))
        char_index = prd.get_vocabulory(word_index)
        char_idxs = prd.get_chars(sentences_t, max_length, char_index)
        label_idxs  = prd.get_sequence_indices(labels_t, label_index, max_length)
        seq_idxs = prd.get_sequence_indices(sentences_t, word_index, max_length)
        assert (len(seq_idxs) == len(label_idxs)),"length of I/O sequences doesn't match"
        
        index_labels = {}
        for item,i in label_index.items():
            index_labels[i] = item 
            
        model_t = mdl.get_model(word_embeddings, max_length, len(char_index), len(index_labels),True)
        # Predict labels for test data
        pred_label = np.asarray(model_t.predict([seq_idxs,char_idxs]))
        pred_label = np.argmax(pred_label,axis=-1)
        #Skipping padded sequences
        pred_label = prd.get_orig_labels(pred_label,index_labels,labels_t)
        print("Predicted Labels--->\n",pred_label[0])
        
        
        outputfile = configs.OUTPUT_FILE
        with open(outputfile, 'w', encoding='utf-8') as f:
            f.write('WORD'+ ' '+ 'TRUE_LABEL'+ ' '+ 'PRED_LABEL')
            f.write('\n')
            f.write('\n')
            for i in range(len(pred_label)):
                cur_sentences = sentences_t[i]
                cur_labels = labels_t[i]
                cur_pred = pred_label[i]
                for j in range(len(cur_sentences)):
                    f.write(cur_sentences[j]+ ' '+ cur_labels[j]+ ' '+ cur_pred[j])
                    f.write('\n')
                f.write('\n')
            f.write('\n')
        
        with open(outputfile, 'r', encoding = 'utf-8') as f:
            for line in f.readlines():
                print(line)

        return ([labels_t, pred_label])
        
                
        #print("Predicted Labels--->\n",pred_label[1])
        #print("Predicted Labels--->\n",pred_label[2])
        #print("Predicted Labels--->\n",pred_label[3])
        
    def evaluation(data, *args, **kwargs):
        labels_t = data[0]
        pred_label = data[1]
        print('Report:\n', classification_report(labels_t, pred_label))
        show = classification_report(labels_t, pred_label)
        return show.replace('\n','').split()[-4:-1]

            
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    