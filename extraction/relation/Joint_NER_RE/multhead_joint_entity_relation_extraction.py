import string
import nltk
from nltk.chunk import  tree2conlltags



import os
class multihead_joint_entity_relation_extraction:
    def read_dataset(self,filename):
        with open(filename) as f:
            data = f.readlines()
        return data


    def data_preprocess(self,data):
        def generate_ne_tags(words_sentences1):
            # takes a tokenized sentence and generates the NE tags for it
            answer = []
            for sentence in words_sentences1:
                result = []
                sent = nltk.pos_tag(sentence)
                pattern = 'NP: {<DT>?<JJ>*<NN>}'
                cp = nltk.RegexpParser(pattern)
                cs = cp.parse(sent)
                iob_tagged = tree2conlltags(cs)
                for i in range(0, len(iob_tagged)):
                    result.append((sentence[i], iob_tagged[i][-1]))
                answer.append(result)
            return answer

        answer=[]
        sentences = []
        e1_start_pos = []
        e1_end_pos = []
        e2_start_pos = []
        e2_end_pos = []
        relations = []
        entity1_in=[]
        entity2_in=[]

        for record in data:
            values = record.split("\t")

            sentences.append(values[0])
            e1_start_pos.append(values[3])
            e1_end_pos.append(values[4])
            e2_start_pos.append(values[7])
            e2_end_pos.append(values[8])
            entity1_in.append(values[1])
            entity2_in.append(values[5])
            relations.append(values[9].strip("\n"))

        for k in range(0,len(sentences)):





            # get the entities
            entity1 = sentences[k][int(e1_start_pos[k]):int(e1_end_pos[k])+1]
            entity2 = sentences[k][int(e2_start_pos[k]):int(e2_end_pos[k])+1]

            relation = relations[k]

            # removes all punctuation
            sentences[k] = sentences[k].translate(str.maketrans('', '', string.punctuation))
            words = sentences[k].split()

            # get which words are involved in the relationship
            for i in range(0, len(words)):
                if words[i] == entity1:
                    pos1 = i
                if words[i] == entity2:
                    pos2 = i


            Classified_text = generate_ne_tags([words])


            for j in range(0,len(words)):

                if words[j] == entity1_in[k] or words[j].startswith(entity1_in[k][:len(words[j])]) :
                    answer.append(str(j) + "\t" + words[j] + "\t" + str(Classified_text[0][j][1]) + "\t" +str([relation]) + "\t"+str([pos2])+"\n")

                else:
                    answer.append(str(j) + "\t" + words[j] + "\t" + str(Classified_text[0][j][1]) + "\t" +str(['N']) + "\t"+str([j])+"\n")


        with open("/home/sanjitha/Downloads/temp_test.txt", 'w+') as f:
            for i in range(0,len(answer)):
                if answer[i].startswith("0"):
                    f.write("#doc"+" "+str(i)+"\n")


                    f.write(answer[i])
                else:
                    f.write(answer[i])



    def train(self):
        script="""
        timestamp=`date "+%d.%m.%Y_%H.%M.%S"`
        output_dir='./logs/'
        config_file='./configs/CoNLL04/bio_config'
    
        # unzip the embeddings file
        unzip data/CoNLL04/vecs.lc.over100freq.zip -d data/CoNLL04/
    
        mkdir -p $output_dir
    
        #train on the training set and evaluate on the dev set to obtain early stopping epoch
        python3 -u train_es.py ${config_file} ${timestamp} ${output_dir} 2>&1 | tee ${output_dir}log.dev_${timestamp}.txt
        """
        os.system("bash -c '%s'" % script)

    def evaluate(self):
        script="""
        timestamp=`date "+%d.%m.%Y_%H.%M.%S"`
        output_dir='./logs/'
        config_file='./configs/CoNLL04/bio_config'
    
        # unzip the embeddings file
        unzip data/CoNLL04/vecs.lc.over100freq.zip -d data/CoNLL04/
    
        mkdir -p $output_dir
    
        python3 -u train_eval.py ${config_file} ${timestamp} ${output_dir} 2>&1 | tee ${output_dir}log.test.${timestamp}.txt
        """
        os.system("bash -c '%s'" % script)