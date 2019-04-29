# Semantic Relation Classification via Bidirectional LSTM Networks with Entity-aware Attention using Latent Entity Typing
Lee, J. and Seo, S. et al, **Semantic Relation Classification via Bidirectional LSTM Networks with Entity-aware Attention using Latent Entity Typing.**, 2019, arXiv:1901.08163
This repository contains the source code for the Relation Extraction presented in the following research publication ([link](https://arxiv.org/pdf/1901.08163.pdf))

### Requirements<br>
* python 3.6
* tensorflow >= 1.6

### Input Data
* SemEval2010
* DDI2013
* NYT

### Input format for training/prediction
* The input format is generalized for the whole Relation group
```
sentence e1 e1_type e1_start_pos e1_end_pos e2 e2_type e2_start_pos e2_end_pos relation (separated by tab)
```
* The output format is generalized for the whole Relation group
```
sentence e1 e2 predicted_relation grandtruth_relation
```

### Sample test data
* ./testexample

### Download word embedding from [here](https://drive.google.com/file/d/1FZj0I7PE7eHbYxVBu95UdQA21-xLVeef/view?usp=sharing) and place ./resource directory.<br>


### Run unit test
```
python sample_test.py
```

### Citation
```
@misc{lee2019semantic,
    title={Semantic Relation Classification via Bidirectional LSTM Networks with Entity-aware Attention using Latent Entity Typing},
    author={Joohong Lee and Sangwoo Seo and Yong Suk Choi},
    year={2019},
    eprint={1901.08163},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
