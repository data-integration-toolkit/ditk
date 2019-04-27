## Test Files

Input format is different from whole group as well as the sample test uploaded on the GitHub

As in Group some needs Knowledge graph as input, some needs Bio data with specialized tag, and sampled test file uploaded on the Github is in CoNLL format whereas this project needs a proper sentence to predict.

For Input it need

- <start_pos>\t<end_pos>\t< sentence>\t<ground_truth>

- Sample Input

  ```
  2	3	California Los Angeles beautiful city	Location
  5	8	He received the 1921 Nobel Prize in Physics for his services to theoretical physics	misc
  2	3	Is New York a beautiful city	Location
  ```



I have updated [sample_test.py](./sample_test.py) code also to match the current test case for columns and rows.



**Note:** Before running main() function directly make sure all files described in [README](../README.md#Input%20format%20for%20training) are present in the **data** directory

