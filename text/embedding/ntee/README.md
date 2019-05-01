# Method Name
- Learning Distributed Representations of Texts and Entities from Knowledge Base.
- Ikuya Yamada, Hiroyuki Shindo, Hideaki Takeda, Yoshiyasu Takefuji, Published in ACL 2017 <https://aclweb.org/anthology/Q17-1028>

## Original Code
https://github.com/studio-ousia/ntee

## Description
- Neural Text-Entity Encoder (NTEE) is a neural network model that learns embeddings (or distributed representations) of texts and Wikipedia entities. Our model places a text and its relevant entities close to each other in a continuous vector space.

## Input and Output
- Prediction Input - a word/text/sentence. Prediction Output - Embedding of the input 
- Training Input - a corpus of texts. Output - Embeddings stored in joblib format 

## Evalution
- SICK <http://clic.cimec.unitn.it/composes/materials/SICK.zip> 
- 0.7144 (pearson) 0.6046 (spearman) 8.6439 (Mean Squared Error)

- STS 2014 <http://alt.qcri.org/semeval2014/task10/data/uploads/sts-en-gs-2014.zip>
- OnWN: 0.7204 (pearson) 0.7443 (spearman) 7.2890 (Mean Squared Error)
- deft-forum: 0.5643 (pearson) 0.5491 (spearman) 5.4748 (Mean Squared Error)
- deft-news: 0.7436 (pearson) 0.6775 (spearman) 6.4844 (Mean Squared Error)
- headlines: 0.6876 (pearson) 0.6246 (spearman) 5.8117 (Mean Squared Error)
- images: 0.8204 (pearson) 0.7671 (spearman) 5.7951 (Mean Squared Error)
- tweet-news: 0.7467 (pearson) 0.6592 (spearman) 7.4580 (Mean Squared Error)

- SEMEVAL 2014 <https://github.com/LavanyaMalladi/SemanticSimilarity/blob/master/Datasets/SemEval%202017/sts-test.csv>
- 0.6949 (pearson) 0.6716 (spearman) 5.6852 (Mean Squared Error)

## Demo
- Link to the Jupyter Notebook 
- Youtube: <https://www.youtube.com/watch?v=DxD6H7SqfjA&t=178s>
