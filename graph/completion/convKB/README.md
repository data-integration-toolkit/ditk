# ConvKB: A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network

## Requirements
- Python 3
- Tensorflow >= 1.6

## Input and Output for Prediction
* input: [entity, relation, entity] triples
* output: [entity, relation, entity] triples

## Usage

### read_data
MUST be called before any other functions

#### required parameters:
* data_name: data file name

#### optional parameter:
* split_ratio: ratio to split the dataset into (train, dev, test)
* embedding_dim: dimensionality of entity and relation embeddings (default=50)
* batch_size: batch size (default=50)


### train

#### required parameters:
* data: dataset for training 

#### optional parameter:
* num_epochs: number of training epochs (default=201)
* save_step: saving step for later loading(default=200)
* dropout_keep_prob: dropout keep probability (default: 1.0).
* l2_reg_lambda: L2 regularizaion lambda (default: 0.001)


### predict

#### required parameters:
* data: dataset for predicting

#### optional parameter:
* model_index: index of loading model (default=200)
* dropout_keep_prob: dropout keep probability (default: 1.0).
* l2_reg_lambda: L2 regularizaion lambda (default: 0.001)


### evaluate

#### required parameters:
* data: dataset for predicting

#### optional parameter:
* model_index: index of loading model (default=200)
* dropout_keep_prob: dropout keep probability (default: 1.0).
* l2_reg_lambda: L2 regularizaion lambda (default: 0.001)

## Benchmarks
* WN18 / WN18RR
* FB15 / FB15k-237

## Evaluation Metrics
* MRR (WN18RR: 0.248 | FB15k-237: 0.396)
* HITS@10 (WN18RR: 52.5 | FB15k-237: 51.7)

## Demo Video
https://youtu.be/vD6bxcFRv_s

# Original Author 
This program provides the implementation of the CNN-based model ConvKB for the knowledge base completion task. ConvKB obtains new state-of-the-art results on two standard datasets: WN18RR and FB15k-237 as described in [the paper](http://www.aclweb.org/anthology/N18-2053):

        @InProceedings{Nguyen2018,
          author={Dai Quoc Nguyen and Tu Dinh Nguyen and Dat Quoc Nguyen and Dinh Phung},
          title={{A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network}},
          booktitle={Proceedings of the 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)},
          year={2018},
          pages={327--333}
          }
  
Please cite the paper whenever ConvKB is used to produce published results or incorporated into other software. I would highly appreciate to have your bug reports, comments and suggestions about ConvKB. As a free open-source implementation, ConvKB is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 

ConvKB is free for non-commercial use and distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA) License. 

<p align="center"> 
<img src="https://github.com/daiquocnguyen/ConvKB/blob/master/model.png" width="344" height="400">
</p>
