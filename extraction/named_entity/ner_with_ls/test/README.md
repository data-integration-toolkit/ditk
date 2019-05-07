NER-with-ls model needs pre-trained embedding for Lexical Simiarlity, word-embedding, Character and Captalized embedding. So I uploaded conll.joblib and ontonotes.joblib under 'data' folder. This is also a reason why I cannot use group common input type; I need to know where the words in input are embedding(conll.joblib OR ontonotes.joblib). <br>

So you can put 2 types of input; **conll2003**, **ontonotes**. <br>

## To run test.py with sample test data
Initial setting is that model will run withconll2003 data and you can change to sample ontonotes dataset in test.py file; I commented out for ontonotes file_path. Plue, Test script will check number of input/output rows and cols. 
Run on the ```ner_with_ls folder``` with following command line; ``` python test/test.py```
