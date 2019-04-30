from MinHash import MinHash

def main(input_file):
    print('Evaluation on SICK DATASET')
    minHash=MinHash()


    '''
        Read the input dataset into a DataFrame : sentence_A, sentence_B, similarity_score

    '''
    input_df= minHash.read_dataset(input_file) 
    predict_similarity_list=[]
    actual_similarity_list=[]

    '''
        Predict the MinHash Signature of the word -- Emebedding

    '''
    input_word = 'kids'
    word_embedding=minHash.generate_embeddings([input_word])
    print(word_embedding)

    '''
        predict_similarity returns the similarity score between the 2 sentences

    '''
    for index, row in input_df.iterrows():
        predicted_similarity_score=minHash.predict(row['sentence_A'],row['sentence_B']) # Predicted value using the Minhaeawq
        actual_similarity_list.append(row['relatedness_score']/5.0) # Retrieved from the Truth values of the datset
        predict_similarity_list.append(predicted_similarity_score) 

    outfile = open("output.txt","w")
    outfile.write("\n".join(str(j) for j in predict_similarity_list))


    ''' 
        Evaluate on SICK Dataset
        - PearsonR Coefficient 
        - SpearmanR Coefficient

    '''
    print('-------------------------------------------------------------')

    print('EVALUATION on Sick Dataset')
    pearsonr, spearmanr =minHash.evaluate(actual_similarity_list,predict_similarity_list)
    print('Pearsonr Coefficient : '+str(pearsonr))
    print('Speearmanr Coefficient :'+str(spearmanr))
    print('-------------------------------------------------------------')


    '''
        Evaluation on other datasets 
    
    '''
    print('-------------------------------------------------------------')
    print('EVALUATION on semEval-2017 Dataset')
    pearsonr_semEval, spearmanr_semEval = minHash.evaluate_util('semEval')
    print('Pearsonr Coefficient : '+str(pearsonr_semEval))
    print('Speearmanr Coefficient :'+str(spearmanr_semEval))

    print('-------------------------------------------------------------')

    print('EVALUATION on semEval-2014 Dataset')
    pearsonr_semEval2014, spearmanr_semEval2014 = minHash.evaluate_util('semEval2014')
    print('Pearsonr Coefficient : '+str(pearsonr_semEval2014))
    print('Speearmanr Coefficient :'+str(spearmanr_semEval2014))

    print('-------------------------------------------------------------')


    


if __name__ == '__main__':
    main("Datasets/sick.xlsx")