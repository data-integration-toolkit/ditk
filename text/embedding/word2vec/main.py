# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:49:52 2019

@author: Lavanya Malladi
"""

from word2vec import Word2Vec_Util
def main(fileName):

    word2vec_util=Word2Vec_Util()

    
    # Load the data for : Training and Testing
    # For this example, I am using SICK Dataset
    
    train_sick_data,test_data_list=word2vec_util.read_Dataset("sick",fileName)

    # Preprocess the data --> Convert it into words

    train_sick_data=word2vec_util.preprocess(train_sick_data)

    # Send the preprocessed data for training and generate the model
    word2vec_util.model=word2vec_util.train(train_sick_data)

    # Saving the model to a given file path

    word2vec_util.save_model('word2vec_model')

    # Loading the model from a given file path

    model = word2vec_util.load_model('word2vec_model')
    # TESTING THE METHODS FOR PREDICTION

    # Predict Embedding takes the input as a list
    # Word = ['word']
    # Sentence = ['word1', 'word2', 'word3'] -- split the sentence on space

    # Word Embedding

    embedding=word2vec_util.predict_embedding(['kids']) 
    print('Word Embedding  :\n ')
    print(embedding)

    # Sentence Embedding

    embedding=word2vec_util.predict_embedding(['The','kids','are', 'playing','outdoors']) 
    print('Sentence Embedding\n : ')
    print(embedding)


    # TESTING THE METHOD FOR COMPUTING THE SEMANTIC SIMILARITY BETWEEN 2 SENTENCES

    sentence1='The kids are playing outdoors'
    sentence2='The kid is playing'
    similarity_score=word2vec_util.predict_similarity(word2vec_util.model,sentence1,sentence2)
    print("\nSimilarity Score is :\n")
    print(similarity_score)

    # Evaluation of the Word2Vec method on SICK Dataset
    # Returns Pearson Correlation Coefficient and Spearman Correlation Coefficient

    print(word2vec_util.evaluate(word2vec_util.model, filename=fileName, evaluation_type='sick'))

    # Verify Predict Embedding

    output_file_path = word2vec_util.verify_predict_embedding(fileName) # fileName == Path for input.txt
    return output_file_path

    
    '''
    To verify the other datasets :
    
    word2vec_util.verify_sick(filename=fileName)
    word2vec_util.verify_semEval(filename=fileName)

    word2vec_util.verify_reviews(filename=fileName)

    '''



if __name__ == '__main__':
    print(main('C:/Users/malla/OneDrive/Documents/548ProjectCode/Word2Vec/Datasets/')) # Path where the datasets have been saved