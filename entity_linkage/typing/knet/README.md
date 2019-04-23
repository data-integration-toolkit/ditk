# Neural Entity Typing with Knowledge Attention

This repo is based on the following paper and Github implementation:

*   Xin, Ji and Lin, Yankai and Liu, Zhiyuan and Sun, Maosong. *Improving neural fine-grained entity typing with knowledge attention.* *Thirty-Second AAAI Conference on Artificial Intelligence*, Hilton New Orleans Riverside, New Orleans, 2018. [pdf](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16321/16167).
*   https://github.com/thunlp/KNET



## How to use our code for KNET

### Prerequisite

*   Actual code on python 2.7.6 [GitHub](https://github.com/thunlp/KNET). This code converted for python 3
*   numpy >=1.13.3
*   tensorflow 0.12.1
    *   Find specific machine TensorFlow version from the follwoing link
        *   https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.12/tensorflow/g3doc/get_started/os_setup.md
    



### To Run

1. Import the Module from main.py

   - ```python
     from main import knet
     ```

2. Create Instanse of the module

   - ```python
     knet_instance = knet()
     ```

3. First call read_dataset() and give all the files required to train in the given order
   - types
   - disamb_file
   - embedding.npy
   - glove
   - Context file for train
   - Entity file for train
   - Fbid for train
   - Label for train
   - Context file to validate
   - Entity file to validate
   - Fbid to validate
   - Label to validate

   - To simply run get the sample files from following link:
     
- https://drive.google.com/open?id=1s59j28nl7mjDxcwhSUBLWn13_hnrLXqL
     
- To run on full train data download files from this two links:
   
     - https://drive.google.com/file/d/1I6h-k2w_ppQ7ASd7wanYfkCfKp0tnSe0/view?usp=sharing
  - http://nlp.stanford.edu/data/glove.840B.300d.zip
   
     ```python
     knet_instance.read_dataset([
         "data/types",
         "data/disamb_file",
         "data/embedding.npy",
         "data/glove.840B.300d.txt",
         "data/train_context.npy",
         "data/train_entity.npy",
         "data/train_fbid.npy",
         "data/train_label.npy",
         "data/valid_context.npy",
         "data/valid_entity.npy",
         "data/valid_fbid.npy",
         "data/valid_label.npy"
     ])
  ```
   
- **Note**: After extracting unzip all the compressed files in it
   
4. To train run the following commands. If already trained you can skip this.

   - ```python
     knet_instance.train(None)
     ```

5. To predict on any sentence give path to file. And each line should contain text in following format:

   - **<start_pos>\t<end_pos>\t< sentence>\t<ground_truth (optional in this case)>**

   - Sample file can be found via this link: https://drive.google.com/file/d/1UI_i4f5ueTN8-Inqg7FESGgSKNbN1ZFG/view?usp=sharing

     ```python
     results = knet_instance.predict(["data/entity_typing_test_input.txt"])
     print(results)
     ```

6. To evaluate on Test Data Provided By the Paper

   - Run the following code

     ```python
     knet_instance.evaluate([
             "data/test_context",
             "data/test_entity",
             "data/test_fbid.npy",
             "data/test_label.txt",
             "data/manual_context.npy",
             "data/manual_entity.npy",
             "data/manual_fbid.npy",
             "data/manual_label.npy"
         ], options={"paper": True})
     ```

   - This files can be found from the following link:
     
     - https://drive.google.com/file/d/1mxP1peepYp03Vr2WES0Hx5axP4npu1CJ/view?usp=sharing

7. Evaluate on provided dataset in the predict (step 5)

   - **Note**: In this case ground_truth must be present, 

     ```python
     precision, recall, f1_score = knet_instance.evaluate([])
     print("precision: {}\trecall: {}\tf1: {}".format(precision, recall, f1_score))
     ```

   - If ground truth was not given in predict give location of new file in the list

     ```python
     precision, recall, f1_score = knet_instance.evaluate(["data/new_file.txt"])
     print("precision: {}\trecall: {}\tf1: {}".format(precision, recall, f1_score))
     ```

     