import csv
import pickle
import gensim
from text2digits import text2digits
from gensim.models import doc2vec
from collections import namedtuple
from gensim.models import Word2Vec
from scipy import spatial
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import scipy
from scipy import stats
from nltk.chunk import  tree2conlltags
import numpy as np
class sensim:

    def read_dataset(self,filename):
        sentences1 = []
        sentences2 = []
        all_sentences = []
        similarity_score = []
        with open(filename, 'r') as f:

            data = csv.reader(f)
            for row in data:

                try:
                    if row[0] != '' and row[1] != '':
                        sentences1.append(row[0])
                        sentences2.append(row[1])
                        all_sentences.append(row[0])
                        all_sentences.append(row[1])

                        similarity_score.append(float(float(row[2]) / 5))
                except:
                    continue

        return sentences1, sentences2, all_sentences, similarity_score


    def train(self,sentences1,sentences2,all_sentences,similarity_score):

        def preprocess(sentences1, sentences2):
            # function to convert it into lower case
            sentences3 = []
            sentences4 = []
            for sentence in sentences1:
                sentences3.append(sentence.lower())
            for sentence in sentences2:
                sentences4.append(sentence.lower())
            return sentences3, sentences4

        sentences1, sentences2 = preprocess(sentences1, sentences2)

        def contractionsEN(sentences):
            # copywrite source: https://github.com/cipriantruica/CATS/blob/master/cats/nlplib/static.py
            # slightly modified dict than the source
            # takes words like he'll and convert it into he will
            # tokenize the input by splitting on spaces
            result = []
            for sentence in sentences:
                sentence = sentence.replace(".,'[]", " ")
                words = sentence.split()

                contractions_en2 = {
                    "'s": " is",
                    "'ve": " have",
                    "'d": " had",
                    "ain't": "am not",
                    "aren't": "are not",
                    "can't": "cannot",
                    "can't've": "cannot have",
                    "'cause": "because",
                    "could've": "could have",
                    "couldn't": "could not",
                    "couldn't've": "could not have",
                    "didn't": "did not",
                    "doesn't": "does not",
                    "don't": "do not",
                    "hadn't": "had not",
                    "hadn't've": "had not have",
                    "hasn't": "has not",
                    "haven't": "have not",
                    "he'd": "he had",
                    "he'd've": "he would have",
                    "he'll": "he will",
                    "he'll've": "he will have",
                    "he's": "he has",
                    "how'd": "how did",
                    "how'd'y": "how do you",
                    "how'll": "how will",
                    "how's": "how has",
                    "i'd": "i had",
                    "i'd've": "i would have",
                    "i'll": "i will",
                    "i'll've": "i will have",
                    "i'm": "i am",
                    "i've": "i have",
                    "isn't": "is not",
                    "it'd": "it had",
                    "it'd've": "it would have",
                    "it'll": "it will",
                    "it'll've": "it will have",
                    "it's": "it is",
                    "let's": "let us",
                    "ma'am": "madam",
                    "mayn't": "may not",
                    "might've": "might have",
                    "mightn't": "might not",
                    "mightn't've": "might not have",
                    "must've": "must have",
                    "mustn't": "must not",
                    "mustn't've": "must not have",
                    "needn't": "need not",
                    "needn't've": "need not have",
                    "o'clock": "of the clock",
                    "oughtn't": "ought not",
                    "oughtn't've": "ought not have",
                    "shan't": "shall not",
                    "sha'n't": "shall not",
                    "shan't've": "shall not have",
                    "she'd": "she had",
                    "she'd've": "she would have",
                    "she'll": "she will",
                    "she'll've": "she will have",
                    "she's": "she is",
                    "should've": "should have",
                    "shouldn't": "should not",
                    "shouldn't've": "should not have",
                    "so've": "so have",
                    "so's": "so is",
                    "that'd": "that had",
                    "that'd've": "that would have",
                    "that's": "that is",
                    "there'd": "there would",
                    "there'd've": "there would have",
                    "there's": "there is",
                    "they'd": "they would",
                    "they'd've": "they would have",
                    "they'll": "they will",
                    "they'll've": "they will have",
                    "they're": "they are",
                    "they've": "they have",
                    "to've": "to have",
                    "wasn't": "was not",
                    "we'd": "we would",
                    "we'd've": "we would have",
                    "we'll": "we will",
                    "we'll've": "we will have",
                    "we're": "we are",
                    "we've": "we have",
                    "weren't": "were not",
                    "what'll": "what will",
                    "what'll've": "what will have",
                    "what're": "what are",
                    "what's": "what is",
                    "what've": "what have",
                    "when's": "when has",
                    "when've": "when have",
                    "where'd": "where did",
                    "where's": "where has",
                    "where've": "where have",
                    "who'll": "who will",
                    "who'll've": "who will have",
                    "who's": "who has",
                    "who've": "who have",
                    "why's": "why has",
                    "why've": "why have",
                    "will've": "will have",
                    "won't": "will not",
                    "won't've": "will not have",
                    "would've": "would have",
                    "wouldn't": "would not",
                    "wouldn't've": "would not have",
                    "y'all": "you all",
                    "y'all'd": "you all would",
                    "y'all'd've": "you all would have",
                    "y'all're": "you all are",
                    "y'all've": "you all have",
                    "you'd": "you had",
                    "you'd've": "you would have",
                    "you'll": "you will",
                    "you'll've": "you will have",
                    "you're": "you are",
                    "you've": "you have"
                }
                words2 = []
                for word in words:
                    try:

                        words2.append(contractions_en2[word])
                    except:
                        words2.append(word)
                sen = ' '.join(i for i in words2)
                result.append(sen)
            return result

        sentences1 = contractionsEN(sentences1)
        sentences2 = contractionsEN(sentences2)

        def tokenize(sentences1, sentences2):
            # splits the sentence into words based on space
            words_sentences1 = []
            words_sentences2 = []
            for sentence in sentences1:
                words_sentences1.append(sentence.split())
            for sentence in sentences2:
                words_sentences2.append(sentence.split())
            return words_sentences1, words_sentences2

        words_sentences1, words_sentences2 = tokenize(sentences1, sentences2)

        # def generate_netags(words_sentences1,words_sentences2):

        def generate_postags(words_sentences1, words_sentences2):
            # takes as input a tokenized sentence and returns their NE tags
            ne_tags_sentences1 = []
            ne_tags_sentences2 = []
            for words in words_sentences1:
                ne_tags_sentences1.append(nltk.pos_tag(words))
            for words in words_sentences2:
                ne_tags_sentences2.append(nltk.pos_tag(words))
            return ne_tags_sentences1, ne_tags_sentences2

        ne_tags_sentences1, ne_tags_sentences2 = generate_postags(words_sentences1, words_sentences2)

        def group_by_tag(ne_tags_sentences1, ne_tags_sentences2):
            res1 = []
            res2 = []
            for words in ne_tags_sentences1:
                tags = set()

                for i in range(0, len(words)):
                    tags.add(words[i][1])
                tags = list(tags)
                sen_tag_pair = {}
                for i in range(0, len(tags)):
                    temp = []
                    for j in range(0, len(words)):
                        if words[j][1] == tags[i]:
                            temp.append(words[j][0])
                    sen_tag_pair[tags[i]] = temp

                res1.append(sen_tag_pair)

            for words in ne_tags_sentences2:
                tags = set()

                for i in range(0, len(words)):
                    tags.add(words[i][1])
                tags = list(tags)
                sen_tag_pair = {}
                for i in range(0, len(tags)):
                    temp = []
                    for j in range(0, len(words)):
                        if words[j][1] == tags[i]:
                            temp.append(words[j][0])
                    sen_tag_pair[tags[i]] = temp

                res2.append(sen_tag_pair)
            return res1, res2

        ne_grouped_sentence1, ne_grouped_sentence2 = group_by_tag(ne_tags_sentences1, ne_tags_sentences2)
        ne_grouped_sentence1 = ne_grouped_sentence1[:5708]

        # print(ne_grouped_sentence1[100])

        def pair_similar_tags(ne_grouped_sentences1, ne_grouped_sentences2):
            # takes the ne tagged sentences and groups together the words that have the same NE tags

            matched_pairs = []
            for i in range(0, len(ne_grouped_sentence1)):
                result = {}
                for key, val in ne_grouped_sentence1[i].items():

                    for key1, val1 in ne_grouped_sentence2[i].items():

                        if key == key1:
                            res = []
                            res.append(val)
                            res.append(val1)
                            result[key] = res
                matched_pairs.append(result)
            return matched_pairs

        matched_pairs = pair_similar_tags(ne_grouped_sentence1, ne_tags_sentences2)

        def bag_of_words(words1, words2):
            # takes as an input two lists which are the words present in 2 sentences and combines them into a single list

            bag = []
            for word in words1:
                bag.append(word)
            for word in words2:
                bag.append(word)
            return bag

        def skipgram_word2vec(bag):
            # takes two sentences(their bag of words representation)
            # converts each word into their word vector representation
            # uses the skipgram model

            model = gensim.models.Word2Vec([bag], min_count=1, window=3, sg=1)
            return model

        # print(words_sentences1[3])
        # print(words_sentences2[3])
        # print(bag_of_words(words_sentences1[3],words_sentences2[3]))
        #
        # print(matched_pairs[3])

        def calculate_similarity(matched_pairs, word_sentences1, word_sentences2):
            # calculate the cosine similarity of the vector representation of 2 sentences
            cosine_similarity = []
            for i in range(0, len(words_sentences1)):
                bag = bag_of_words(word_sentences1[i], word_sentences2[i])
                model = skipgram_word2vec(bag)

                final = 0
                match = 0
                for key, val in matched_pairs[i].items():

                    first_list = val[0]
                    second_list = val[1]

                    result = 0
                    if len(first_list) > len(second_list):
                        match += len(second_list)
                        for word1 in second_list:
                            max = -100
                            for word2 in first_list:
                                similar = model.similarity(word1, word2)
                                if similar > max:
                                    max = similar

                            result += max


                    else:
                        match += len(first_list)
                        for word1 in first_list:
                            max = -100
                            for word2 in second_list:
                                similar = model.similarity(word1, word2)
                                if similar > max:
                                    max = similar

                            result += max

                    final += result
                try:
                    temp1 = final / match
                    cosine_similarity.append(temp1)
                except:
                    cosine_similarity.append(0)

            return cosine_similarity

        cos_sim = calculate_similarity(matched_pairs, words_sentences1, words_sentences2)

        # print(calculate_similarity(matched_pairs,words_sentences1,words_sentences2))

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

        ne_tagged_sentences1 = generate_ne_tags(words_sentences1)
        ne_tagged_sentences2 = generate_ne_tags(words_sentences2)

        def group_ne_tagged(ne_tagged_sentences1):
            answer = []

            for i in range(0, len(ne_tagged_sentences1)):
                tags = set()
                for j in range(0, len(ne_tagged_sentences1[i])):
                    tags.add(ne_tagged_sentences1[i][j][1])
                tags = list(tags)
                final_tag = {}
                for tag in tags:
                    result = []
                    for j in range(0, len(ne_tagged_sentences1[i])):
                        if ne_tagged_sentences1[i][j][1] == tag:
                            result.append(ne_tagged_sentences1[i][j][0])
                    final_tag[tag] = result
                answer.append(final_tag)

            return answer

        grouped_sentence1 = group_ne_tagged(ne_tagged_sentences1)
        grouped_sentence2 = group_ne_tagged(ne_tagged_sentences2)

        matched_pairs2 = pair_similar_tags(grouped_sentence1, grouped_sentence2)
        cos_sim2 = calculate_similarity(matched_pairs2, words_sentences1, words_sentences2)

        def bow_vec(sentences1, sentences2):
            # takes two sentences(their bag of words representation)
            # converts each word into their word vector representation
            # uses the continuous bag of words model

            docs = []
            analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
            for i, text in enumerate(sentences1):
                words = text.lower().split()
                tags = [i]
                docs.append(analyzedDocument(words, tags))
            model11 = doc2vec.Doc2Vec(docs, vector_size=50, min_count=1)

            docs2 = []
            analyzedDocument2 = namedtuple('AnalyzedDocument', 'words tags')
            for i, text in enumerate(sentences2):
                words = text.lower().split()
                tags = [i]
                docs2.append(analyzedDocument2(words, tags))
            model22 = doc2vec.Doc2Vec(docs2, vector_size=50, min_count=1)
            return model11, model22

        model11, model22 = bow_vec(sentences1, sentences2)

        def written_num():
            # program to convert thirty five to 35

            converter = text2digits.Text2Digits()
            num_similarity = []
            for i in range(0, len(sentences1)):
                sen1 = converter.convert(sentences1[i])
                sen2 = converter.convert(sentences2[i])
                num1 = 0
                num2 = 0
                for s in sen1.split():
                    if s.isdigit():
                        num1 += int(s)
                for s in sen2.split():
                    if s.isdigit():
                        num2 += int(s)

                if num1 != 0 and num2 != 0:
                    num_similarity.append(min(num1, num2) / max(num1, num2))
                else:
                    num_similarity.append(0)

            return num_similarity

        num_sim = written_num()

        # print(written_num())

        def length_sentences(sentences1, sentences2):
            diff = []
            for i in range(0, len(sentences1)):
                diff.append(abs(len(sentences1[i]) - len(sentences2[i])))
            return diff

        diff = length_sentences(sentences1, sentences2)

        def bow_similarity():
            # input: sentences
            # output: tf-idf weighted vector of the sentence

            all_sentences.append(" ")
            model = Word2Vec(all_sentences, min_count=1, size=100)
            vocab = model.wv.vocab.keys()
            wordsInVocab = len(vocab)

            import numpy as np

            def sent_vectorizer(sent, model):
                sent_vec = np.zeros(100)
                numw = 0
                for w in sent:
                    try:
                        sent_vec = np.add(sent_vec, model[w])
                        numw += 1
                    except:
                        pass
                return sent_vec / np.sqrt(sent_vec.dot(sent_vec))

            V = []
            for sentence in all_sentences:
                V.append(sent_vectorizer(sentence, model))

            from numpy import dot
            from numpy.linalg import norm

            results = [[0 for i in range(len(V))] for j in range(len(V))]

            for i in range(len(V) - 1):
                for j in range(i + 1, len(V)):
                    results[i][j] = dot(V[i], V[j]) / norm(V[i]) / norm(V[j])

            # print(results)

            from sklearn.feature_extraction.text import TfidfTransformer
            tfidf = TfidfTransformer(norm="l2")
            tfidf.fit(results)
            # print("IDF:", tfidf.idf_)
            tf_idf_matrix = tfidf.transform(results)

            answer = tf_idf_matrix.todense()

            sparse_similarity = []
            evens = [x for x in range(len(all_sentences)) if x % 2 == 0]
            odds = [x for x in range(len(all_sentences)) if x % 2 != 0]

            for z in range(0, len(evens)):
                try:
                    sparse_similarity.append(1 - spatial.distance.cosine(answer[evens[z]], answer[odds[z]]))
                except:
                    continue

            return sparse_similarity

        xyz = bow_similarity()

        with open("C:\\Users\\sanji\\Desktop\\training_features_v1.csv", 'w+', encoding='utf-8') as g:
            for i in range(0, len(sentences1)):


                # feature 1
                g.write(str(cos_sim[i]) + "\t")

                # features 2

                g.write(str(cos_sim2[i]) + "\t")

                # feature 3

                g.write(str(xyz[i]) + "\t")

                # feature 4

                g.write(str(num_sim[i]) + "\t")

                # for feature 5
                g.write(str(diff[i]) + "\t")

                # actual  similarity value
                g.write(str(similarity_score[i]) + "\t")

                g.write("\n")
        df = pd.read_csv('C:\\Users\\sanji\\Desktop\\training_features_v1.csv', header=None, sep='\t')
        df.rename(columns={0: 'ne_tagged', 1: 'pos_tagged', 2: 'bag_of_words', 3: 'numeric_sum', 4: 'diff_in_length',5: 'actual_similarity'}, inplace=True)
        df.to_csv('C:\\Users\\sanji\\Desktop\\training_features.csv', index=False)



    def save_model(self,rf):



        pickle.dump(rf, open("C:\\Users\\sanji\\Desktop\\finalized_model.sav", 'wb'))

    def load_model(self):
        loaded_model = pickle.load(open("C:\\Users\\sanji\\Desktop\\finalized_model.sav", 'rb'))
        return loaded_model

    def evaluate(self,path):
        dataset = pd.read_csv(path)

        X = dataset.drop('actual_similarity', axis=1)
        y = dataset['actual_similarity']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        X_train = X_train.values
        X_train = np.nan_to_num(X_train)
        y_train = y_train.values
        y_train = np.nan_to_num(y_train)
        X_test = np.nan_to_num(X_test.values)
        y_test = np.nan_to_num(y_test.values)
        rf = RandomForestRegressor(n_estimators=1024, random_state=48, max_depth=8)
        rf.fit(X_train, y_train)
        self.save_model(rf)
        model=self.load_model()

        predictions = model.predict(X_test)

        a = scipy.stats.pearsonr(predictions, y_test)
        print("The pearson correlation Coefficient is ", a[0].__round__(6) * 100)

        return a[0].__round__(6) * 100



