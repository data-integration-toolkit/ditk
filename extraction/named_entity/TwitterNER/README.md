[![Build Status](https://travis-ci.org/dbaumgarten/FToDTF.svg?branch=master)](https://travis-ci.org/dbaumgarten/FToDTF)
[![Codacy Badge](https://api.codacy.com/project/badge/Coverage/3872f2d4f965425ea150abd921027f4c)](https://www.codacy.com/app/incognym/FToDTF?utm_source=github.com&utm_medium=referral&utm_content=dbaumgarten/FToDTF&utm_campaign=Badge_Coverage)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/3872f2d4f965425ea150abd921027f4c)](https://www.codacy.com/app/incognym/FToDTF?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=dbaumgarten/FToDTF&amp;utm_campaign=Badge_Grade)
# FToDTF - FastText on Distributed TensorFlow

This software uses unsupervised machine-learning to calculate vector-representation of words. These vector representations can then be used for things like computing the similarity of words to each other or association-rules (e.g. paris is to france like X to germany).

This software is an implementation of https://arxiv.org/abs/1607.04606 (facebook's fasttext) in tensorflow.  

In contrast to the original implementation of fasttext (https://github.com/facebookresearch/fastText) this implementation can use GPUs to accelerate the training and the training can be distributed across multiple nodes.

## Installation
Run ```pip3 install git+https://github.com/dbaumgarten/FToDTF.git```  
The programm is now installed system-wide. You can now import the package ftodtf in python3 and run the cli-command ```fasttext <optional args>```

## Running
After installing just run  
```
fasttext preprocess --corpus_path <your-training-data>
fasttext train
```  
in your console.  
This will run the training and will periodically store checkpoints of the current model into the ./log folder.
After you have trained for some time you can try out the trained word-vectors:
```
fasttext infer similarity i you one two
```
This will load the latest model stored in ./log and use it to calculate and print the similarity between the words i you one two. If everything works out, "I" should be similar to "you" and "one" should be similar to "two", while all other combinations should be relatively un-similar.

## Docker
This application is also available as pre-built docker-image (https://hub.docker.com/r/dbaumgarten/ftodtf/)
```
sudo docker run --rm -it -v `pwd`:/data dbaumgarten/ftodtf train
```

## Distributed Setup
### Docker
There is docker-compose file demonstrating the distributed setup op this programm. To run a cluster on your local machine 
- go to the directory of the docker-compose file
- preprocess your data using `fasttext preprocess --corpus_path <your-training-data>`
- run:
```
sudo docker-compose up
```
This will start a cluster consisting of two workers and two parameter servers on your machine.  
Each time you restart the cluster it will continue to work from the last checkpoint. If you want to start from zero delete the contents of ./log/distributed on the server of worker0
Please note that running a cluster on a single machine is slower then running a single instance directly on this machine. To see some speedup you will need to use multiple independent machines.
### Slurm
There is also an example how to use slurm for setting up distributed training (slurmjob.sh). You will probably have to modify the script to work on your specfic cluster. Please not that the slurm-script currently only handles training. You will have to create training-batches (fasttext preprocess) and copy the created batches-files to the cluster-nodes manually befor starting training.

## Training-data
The input for the proprocess-step is a raw text-file containing lots of sentences of the language for that you want to compute word-embeddings.

## Hyperparameters and Quality
The quality of the calculated word-vectors depends heavily on the used training-corpus and the hyperparameters (training-steps, embedding-dimension etc.). If you don't get usefull results try changing the default hyperparameters (especially the amount of training-steps can have a big influence) or use other training data.  

We got really good results for german with 81MB of training-data and the parameters --num_buckets=2000000 --vocabulary_size=200000 --steps=10000000, but the resulting model is quite large (2.5GB) and it took >10 hours to train.

## Known Bugs and Limitations
- When supplying input-text that does not contain sentences (but instead just a bunch of words without punctuation) ```fasttext preprocess``` will hang indefinetly.

## Documentation
You can find the auto-genrated documentation for the code here: https://dbaumgarten.github.io/FToDTF/  
The architecture documentation (german only) can be found here: https://github.com/dbaumgarten/FToDTF/blob/master/docs/architecture/architecture.md

## Acknowledgements
Computations for this work (like the testing of the distributed-training functionalities) were done with resources of Leipzig University Computing Centre.
