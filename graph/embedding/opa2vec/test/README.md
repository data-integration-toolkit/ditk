The format of the test file is similar to one present in test branch. Minor changes are metrics, I have implmented only cosine similarity measure
Rest of the test cases remain the same
The input format is:
Ontolgy file and association file

The output format is list of vectors representing embedding

The evaluation metric is cosine similarity

The test cases are:

1. The read data method returns the ontology and association files
2. The learn_embedding methods returns a vector of size 296*200
3. The evlaute method returns a dict with cosine similarity

