## Test Files

- The input format is generalized for the whole NER group

  - It contains multiple columns got from CoNLL 2003 and CoNLL 2012 (separated by space)

  - Sample

  - ```
    Yes UH (TOP(S(INTJ*) O bc/cnn/00/cnn_0003 0 0 - - - Linda_Hamilton * -
    they PRP (NP*) O bc/cnn/00/cnn_0003 0 1 - - - Linda_Hamilton * (15)
    did VBD (VP*) O bc/cnn/00/cnn_0003 0 2 do 01 - Linda_Hamilton (V*) -
    ```



I have updated [sample_test.py](./sample_test.py) code also to match the current test case.

To run the unittest, run from the current directory

<code>python sample_test.py</code>

**Note:** Before running main() function directly make sure all files described in [README](../README.md#Input%20format%20for%20training) are present in the **data** directory