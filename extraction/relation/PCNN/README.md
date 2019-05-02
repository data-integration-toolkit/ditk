# PCNN

Refactored by Xiangci Li, xiangcil@usc.edu
Video available [here](https://youtu.be/3J3xF-tM6uY)

## Description
A relation extraction tool proposed by [*Zeng, D., Liu, K., Chen, Y., & Zhao, J. (2015). Distant supervision for relation extraction via piecewise convolutional neural networks. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (pp. 1753-1762).
*](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP203.pdf). Original code available [here](https://github.com/thunlp/OpenNRE).

### Quick Idea
Uses CNN and a novel piecewise pooling to perform feature extraction.

## Requirements
* Python 2.7
* Tensorflow
* scikit-learn

## Usage
* Follow `PCNN.ipynb` for usage.
* You must split the dataset in the common format defined by `ditk.extraction.relation` into `PCNN/data/DATASET_NAME/trainfile.txt`, `PCNN/data/DATASET_NAME/testfile.txt` and run the code in `PCNN.ipynb` sequentially.
* All functions read data from files and store results back to files.
* Run `testPCNN.py` for test.