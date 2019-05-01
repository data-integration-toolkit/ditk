RESIDE: Improving Distantly-Supervised Neural Relation Extraction using Side Information
This repo cites:
•	@InProceedings{reside2018,
  author = 	"Vashishth, Shikhar
		and Joshi, Rishabh
		and Prayaga, Sai Suman
		and Bhattacharyya, Chiranjib
		and Talukdar, Partha",
  title = 	"{RESIDE}: Improving Distantly-Supervised Neural Relation Extraction using Side Information",
  booktitle = 	"Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"1257--1266",
  location = 	"Brussels, Belgium",
  url = 	"http://aclweb.org/anthology/D18-1157"
}

•	Github link to original code: https://github.com/malllabiisc/RESIDE

Dependencies/Prerequisites:
•	Compatible with TensorFlow 1.x and Python 3.x.
•	Dependencies can be installed using requirements.txt.


Description:
Distantly-supervised Relation Extraction (RE) methods train an extractor by automatically
aligning relation instances in a Knowledge Base (KB) with unstructured text. In addition to relation instances, KBs often contain other relevant side information, such as aliases
of relations (e.g., founded and co-founded are aliases for the relation founderOfCompany).
RE models usually ignore such readily available side information. This paper proposes RESIDE, a distantly-supervised neural relation extraction method which utilizes additional side information from KBs for improved relation extraction. It uses entity type
and relation alias information for imposing soft constraints while predicting relations. RESIDE employs Graph Convolution Networks (GCN) to encode syntactic information from
text and improves performance even when limited side information is available.

 

Overview of RESIDE (proposed method): RESIDE first encodes each sentence in the bag by concatenating embeddings (denoted by ⊕) from Bi-GRU and Syntactic GCN for each token, followed by word attention. Then, sentence embedding is concatenated with relation alias information, which comes from the Side Information Acquisition Section, before computing attention over sentences. Finally, bag representation with entity type information is fed to a softmax classifier. Please refer to paper for more details

Side information acquisition:
 

Relation alias side information extraction for a given sentence. First, Syntactic Context Extractor identifies relevant relation phrases P between target entities. They are then matched in the embedding space with the extended set of relation aliases R from KB. Finally, the relation embedding corresponding to the closest alias is taken as relation alias information.



Input and Output:
•	The input data for training as well as testing is in the form of a pickle (.pkl) format. The structure of the processed input data is as follows.
{
    "voc2id":   {"w1": 0, "w2": 1, ...},
    "type2id":  {"type1": 0, "type2": 1 ...},
    "rel2id":   {"NA": 0, "/location/neighborhood/neighborhood_of": 1, ...}
    "max_pos": 123,
    "train": [
        {
            "X":        [[s1_w1, s1_w2, ...], [s2_w1, s2_w2, ...], ...],
            "Y":        [bag_label],
            "Pos1":     [[s1_p1_1, sent1_p1_2, ...], [s2_p1_1, s2_p1_2, ...], ...],
            "Pos2":     [[s1_p2_1, sent1_p2_2, ...], [s2_p2_1, s2_p2_2, ...], ...],
            "SubPos":   [s1_sub, s2_sub, ...],
            "ObjPos":   [s1_obj, s2_obj, ...],
            "SubType":  [s1_subType, s2_subType, ...],
            "ObjType":  [s1_objType, s2_objType, ...],
            "ProbY":    [[s1_rel_alias1, s1_rel_alias2, ...], [s2_rel_alias1, ... ], ...]
            "DepEdges": [[s1_dep_edges], [s2_dep_edges] ...]
        },
        {}, ...
    ],
    "test":  { same as "train"},
    "valid": { same as "train"},
}
o	voc2id is the mapping of word to its id
o	type2id is the maping of entity type to its id.
o	rel2id is the mapping of relation to its id.
o	max_pos is the maximum position to consider for positional embeddings.
o	Each entry of train, test and valid is a bag of sentences, where
	X denotes the sentences in bag as the list of list of word indices.
	Y is the relation expressed by the sentences in the bag.
	Pos1 and Pos2 are position of each word in sentences wrt to target entity 1 and entity 2.
	SubPos and ObjPos contains the position of the target entity 1 and entity 2 in each sentence.
	SubType and ObjType contains the target entity 1 and entity 2 type information obtained from KG.
	ProbY is the relation alias side information (refer paper) for the bag.
	DepEdges is the edgelist of dependency parse for each sentence (required for GCN).


•	The output is a mapping of the actual relation to its id along with its predicted relation to its id (the relation to id mapping has been stored in a separate file). It also gives the test accuracy, precision, recall and F1 scores as the output. 


Benchmark Datasets:
•	Riedel NYT 
•	Google IISc Distant Supervision (GIDS) 

Evaluation results:

Dataset	Metrics
	Accuracy	Precision	Recall	F1
NYT	1.00	0.38	0.53	0.44
GIDS	0.93	0.35	0.51	0.41

 

P@N for relation extraction using variable number of sentences in bags (with more than one sentence) in Riedel dataset. Here, One, Two and All represents the number of sentences randomly selected from a bag. RESIDE attains improved precision in all settings.

How to run:
Preprocessing a new dataset:
•	preproc directory contains code for getting a new dataset in the required format (riedel_processed.pkl) for reside.py.
•	Get the data in the same format as followed in riedel_raw for Riedel NYT dataset.
•	Finally, run the script preprocess.sh. make_bags.py is used for generating bags from sentence. generate_pickle.py is for converting the data in the required pickle format.
Evaluating a pretrained model:
The pretrained model has been stored in the checkpoint folder. Run the following command:
python3 reside.py –restore –only_eval
Training from scratch:
Run the command:
python3 reside.py 
Jupyter Notebook:
reside_demo.ipynb
Note: Since RESIDE takes a long time to train, the Jupyter notebook has been run on a model previously trained by me.
YouTube link:




 










