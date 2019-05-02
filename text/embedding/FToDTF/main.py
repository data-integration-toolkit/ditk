from fasttext import FastText
import tensorflow as tf
import os

def main():
    tf.logging.set_verbosity(tf.logging.ERROR)
    dirpath = os.getcwd()
    datapath = dirpath + "/data/sick.xls"
    modelpath = dirpath + "/models/log_sick"
    
    #For SICK    
    fasttext_obj = FastText(input_corpus_path=datapath, modelpath = modelpath)

    # Call read_dataset of corresponding method
    train_data_path,test_data_path = fasttext_obj.read_Dataset(datapath,'sick')

    # Call train
    model, dict_map = fasttext_obj.train(train_data_path)

    # Call Predict
    pred_embedding = fasttext_obj.predict_embedding(['kettle'],model, dict_map)
    print ('Predicted embedding is',pred_embedding)

    similarity = fasttext_obj.predict_similarity('Two dogs are fighting','Two dogs',model, dict_map)
    print ('Similarity is', similarity[0][0])

    # Call evaluate
    eval_score, mean_score,spearman_score = fasttext_obj.evaluate(model,test_data_path,'sick',dict_map)
    print ('Pearson Coefficient is', eval_score)
    print ('Mean Square error is', mean_score)
    print ('Spearman Score is', spearman_score)
    #print (eval_score, mean_score,spearman_score)

if __name__ == '__main__':
    main()
