Neural Collective Entity Linking (NCEL)
Yixin Cao, Lei Hou, Juanzi Li, Zhiyuan Liu-
Proceedings of the 27th International Conference on Computational Linguistics 
https://www.aclweb.org/anthology/C18-1057

Input:
Text,Candidate set

Output:
Probability that entity refers to its mention

Approach:
It combines local and global approach for Entity Linking. Local approach is based on proximity of mentions in text. Global approach is based on topical coherence. The 3 tasks performed are :
1. Generating Canddate set
2. Feature Extraction
3. Neural Model

Candidate Generation: Use a pre-built dictionary to generate set of entities as candidates to be disambiguated for each mention.  E.g., for mention England, we have Φ(mi) = {e i 1 , ei 2 , ei 3 }, in which the entities refer to England national football team, England and England cricket team
Feature Extraction: based on the document and its entity graph, we extract both local features and global features for each candidate entity to feed our neural model. Concretely, local features reflect the compatibility between a candidate and its mention within the contexts, and global features are to capture the topical coherence among various mentions. These features, including vectorial representations of candidates and a subgraph indicating their relatedness, are highly discriminative for tackling ambiguity in EL
Neural Model: we first encode the features to represent nodes (i.e., candidates) in the graph.then improve them for disambiguation via multiple graph convolutions by exploiting the structure information, in which the features for correct candidates that are strongly connected (i.e., topical coherent) shall enhance each other, and features for incorrect candidates are weakened due to their sparse relations. Then, we decode features of nodes to output indicating how possible the candidate refers to its mention

 
	Fig showing the entire process which consists of 3 steps as mentioned above.

Environment:
Python 3.6
Numpy
Gflags
Pyxdameraulevenshtein
Protocol Buffer

Execute the code:  
python main.py --training_data DATASET:SUPPLEMENT:TEXTPATH:MENTIONPATH --eval_data DATASET:SUPPLEMENT:TEXTPATH:MENTIONPATH --candidates_file ncel:CANDIDATE_FILE --wiki_entity_vocab ENTITY_VOCAB --word_embedding_file WORD_VECTOR --entity_embedding_file ENTITY_VECTOR --log_path PATH_TO_LOG


Benchmark Datasets:
CoNLL 2003
AQUAINT
ACE 2004
WW
TAC2010

We couldn’t use AQUAINT and ACE 2004 because it was not publicly accessible to us.

Evaulation Metrics:
F1 score (Micro and Macro)


Results: 
Datasets	F1(Micro)	F1(Macro)
CoNLL 	71	66
WW	72	68
TAC2010	87	84


Link to Youtube video:  https://youtu.be/mL0dXa755r8


