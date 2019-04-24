# Structured Gradient Tree Boosting

## Requirements
* python3
* scikit-learn(0.19.1)
* numpy (1.15.0)

## Data

The preprocessed [AIDA-CoNLL](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/aida/downloads/)
data ('AIDA-PPR-processed-sample.json') is available in the [data](data) folder:
* The entity candidates are generated based on the [PPRforNED](https://github.com/masha-p/PPRforNED) candidate
generation system.
* The system uses 19 local features, including 3 prior features, 4 NER features,
2 entity popularity features, 4 entity type features, and 6 context features. 
Please look into the paper for details.
* The format is:(mentioned_entity, offset_pairs, highest_rank_candidate, set of (candidate_entity, label, features))

The system also uses entity-entity features, which can be quickly computed
on-the-fly. Here, we provide pre-computed entity-entity features (3 features
per entity pair) for the AIDA-CoNLL dataset, which is available in the 
[data](data) folder ('ent_ent_feats.txt.gz').

## Usage

### read_data
* required parameters:
- dataset: data file name
- split_ratio: ratio to split the dataset into (train, dev, test)
* optional parameter:
- num_candidate: number of entity candidates for a mention

### train

### predict

### evaluate

### save_model

### load_model

### Benchmarks
* AIDA-CoNLL
* AQUANT
* ACE

### Evaluation Metrics
* Accuracy
* F1 (in-progress)


## Original Author
Author: Yi Yang

Contact: yyang464@bloomberg.net


    Yi Yang, Ozan Irsoy, and Kazi Shefaet Rahman 
    "Collective Entity Disambiguation with Structured Gradient Tree Boosting"
    NAACL 2018

[[pdf]](https://arxiv.org/pdf/1802.10229.pdf)

BibTeX

    @inproceedings{yang2018collective,
      title={Collective Entity Disambiguation with Structured Gradient Tree Boosting},
      author={Yang, Yi and Irsoy, Ozan and Rahman, Kazi Shefaet},
      booktitle={Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
      volume={1},
      pages={777--786},
      year={2018}
    }
