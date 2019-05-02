This uses only the test_evaluate and test_readdataset methods, for test_predict the test sample looks for output, while my program writes its output into a file, and in a unique format.
test_output_facts doesn't make since for my program because the read dataset method creates a directory containing modified versions of the original file.


Notes: to run the tests you must include the cpp program transX explained in the main README.
To run the evaluation test first we must train our model, and then predict as well which takes some time.
