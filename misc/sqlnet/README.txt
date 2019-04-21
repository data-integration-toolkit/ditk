SQLNet sample_test.py was written by me, it tests basic file loading, db loading, and output types

The input file and output file are identical because the json file includes ground truth and orginal query. 
This avoids tedious indexing and record matching

Upon successful operation, the ground truth json will be regenerated using results from the model

Since the file format contains only 1 input and 1 output file. The glove embedding and test_DB used for running
the test has been omitted