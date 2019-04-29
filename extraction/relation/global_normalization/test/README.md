## Global Normalization Test Cases.

* *Note*
    * *The model uses CRF to globally normalized named entity pair and relation triplet at the output layer.*
    * *It selects the best epoch(Best average f1 score) to evaluate the text file.*
    * *The train module itself predicts the relations for the test file and calculates the evaluation metric(Precision, Recall and F1).*
    * *No model is saved in this process.*
    
* Test Cases Format
    * Read the output file created in the train model.
    * Test the actual and predicted relation in the file.