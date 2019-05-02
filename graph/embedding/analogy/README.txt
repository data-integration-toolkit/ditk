CSCI 548 Project README For Grading/Running
** *See the README.md file in test directory for expanded markdown version of this file

Note on input files:
Most file inputs have been hardcoded at the top of each file (i.e. in main.py, analogy_test.py, analogy_notebook.ipynb).
Those can be changed depending on where the source data is stored.

Unit Test (/test/analogy_test.py):
Unit tests follow input file note from above. I have followed the signatures of the group test with a few minor
exceptions. I replaced the setUp method with setUpClass that reads in the input files and trains the model. Since unit
tests can run in any order, the setUpClass method must read the data and train the model that will be tested by
the other methods in the test class. The tests can be run for either FB15k or WN18 data sets, the assumption for which
can be set at top of file.

Main (/test/main.py):
In my implementation main.py reads the test input file and outputs a file with the embeddings of the test subject,
relation, object triple. The difference between mine and group test input file is I also put the assumptions for
where to read the file in from on the first line.

i.e.
<input_file_dir>,<train_file>,<validate_file>,<whole_text_file>,<test_file>,<relation_file>,<entity_file>,<epochs>,<dimensions>,<output_file>
<subject1>, <relation1>, <object1>
<subject2>, <relation2>, <object2>
...

Note on file paths when running the code (Module Not Found Error):
Since this code is part of a larger group repository, correct paths need to be set in order for modules to be found
and loaded correctly. PyCharm is pretty good about setting paths appropriately, but if running from command line or
Jupyter Notebook the paths may need to be set or you may receive a "Module Not Found Error"


