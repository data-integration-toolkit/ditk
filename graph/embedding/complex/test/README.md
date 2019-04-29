#### CSCI 548 Project README For Grading/Running

**Note on input files**:
Most file inputs have been hardcoded at the top of each file (i.e. in main.py, complex_test.py, complex_notebook.ipynb).
Those can be changed depending on where the source data is stored.
---
##### Unit Test (/test/complex_test.py)

Unit tests follow input file note from above. I have followed the signatures of the group test with a few minor
exceptions. I replaced the setUp method with setUpClass that reads in the input files and trains the model. Since unit
tests can run in any order, the setUpClass method must read the data and train the model that will be tested by
the other methods in the test class. The tests can be run for either FB15k or WN18 data sets, the assumption for which
can be set at top of file.

---

##### Main (/test/main.py)

In my implementation main.py reads the test input file and outputs a file with the embeddings of the test subject,
relation, object triple. The difference between mine and group test input file is I also put the assumptions for
where to read the file in from on the first line.

Input file lines and fields:
```text
<input_file_dir>,<train_file>,<validate_file>,<whole_text_file>,<test_file>,<relation_file>,<entity_file>,<epochs>,<dimensions>,<output_file>
<subject1>, <relation1>, <object1>
<subject2>, <relation2>, <object2>
...
```

Example input file:
```text
D:\USC\CS548\groupdat\WN18\,train.txt,"","","test.txt",relation2id.txt,entity2id.txt,2,20,D:\USC\CS548\ditk\graph\embedding\complex\test\G9_embedding_test_output_WN18.txt
06845599	_member_of_domain_usage	03754979
00789448	_verb_group	01062739
10217831	_hyponym	10682169
```
Example output file:
```text
06845599: [ 0.09501449 -0.12043544 -0.22431554  0.07272141  0.27072931  0.25192355
 -0.28213659 -0.16739177  0.05950536  0.17705623 -0.24862748  0.00077159
 -0.36271339 -0.28749112 -0.28533121  0.04315033  0.2169928   0.02267169
 -0.10473549  0.18355116]
_member_of_domain_usage: [-0.09369002  0.31331353  0.11798782 -0.2479869   0.16487534 -0.16556023
  0.07935038 -0.61014314 -0.35374827  0.00420813 -0.30959673  0.23647503
  0.27329918  0.13721187  0.27686096 -0.02340155  0.00937644  0.08294171
  0.39585813 -0.34677271]
03754979: [-0.11262972  0.07847286 -0.40865625  0.07520694 -0.38941376  0.05658019
 -0.15026309 -0.09810015  0.12451178 -0.5086426   0.26208278 -0.47067987
  0.26734952 -0.42613524 -0.17918553  0.15090269  0.26891225 -0.28180002
  0.34760165  0.21867973]
```

---

*Note on file paths when running the code (Module Not Found Error):*

Since this code is part of a larger group repository, correct paths need to be set in order for modules to be found
and loaded correctly. PyCharm is pretty good about setting paths appropriately, but if running from command line or
Jupyter Notebook the paths may need to be set or you may receive a "Module Not Found Error"