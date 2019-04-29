### CSCI 548 Project README For Grading/Running

**Note on input files**:

Most file inputs have been hardcoded at the top of each file (i.e. in main.py, analogy_test.py, analogy_notebook.ipynb).
Those can be changed depending on where the source data is stored.
---
#### Unit Test (/test/analogy_test.py)

Unit tests follow input file note from above. I have followed the signatures of the group test with a few minor
exceptions. I replaced the setUp method with setUpClass that reads in the input files and trains the model. Since unit
tests can run in any order, the setUpClass method must read the data and train the model that will be tested by
the other methods in the test class. The tests can be run for either FB15k or WN18 data sets, the assumption for which
can be set at top of file.

---

#### Main (/test/main.py)

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
D:\USC\CS548\groupdat\FB15k\,train.txt,"","","test.txt",relation2id.txt,entity2id.txt,2,20,D:\USC\CS548\ditk\graph\embedding\analogy\test\G9_embedding_test_output_FB15k.txt
/m/0bwfn	/education/educational_institution/students_graduates./education/education/major_field_of_study	/m/02h40lc
/m/026c1	/film/actor/film./film/performance/film	/m/03s6l2
```
Example output file:
```text
/m/0bwfn: [-0.32588493  0.18673964 -0.02334621 -0.35974039 -0.12791498 -0.41783713
  0.10942638 -0.18001561  0.45587884 -0.09097599]
/education/educational_institution/students_graduates./education/education/major_field_of_study: [ 0.77041321 -0.16407499 -0.53455424  0.2073073  -0.28994511 -0.14453066
  0.18151202 -0.80410189  0.11575487  0.4431037 ]
/m/02h40lc: [-0.18416133 -0.16373736 -0.14962522 -0.15455913  0.44968564  0.19005451
  0.20788976  0.56292062  0.04254181 -0.04675763]
```

---

*Note on file paths when running the code (Module Not Found Error):*

Since this code is part of a larger group repository, correct paths need to be set in order for modules to be found
and loaded correctly. PyCharm is pretty good about setting paths appropriately, but if running from command line or
Jupyter Notebook the paths may need to be set or you may receive a "Module Not Found Error"