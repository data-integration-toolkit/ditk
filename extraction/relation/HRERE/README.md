# **HRERE : Connecting Language and Knowledge with Heterogeneous Representations for Neural Relation Extraction**
**This repo cites the work of:**

Xu, Peng and Barbosa, Denilson, Connecting Language and Knowledge with Heterogeneous Representations for Neural Relation Extraction, Proceedings of the The 17th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL 2019). Link to the paper:  

•	**Github link to original code**: https://github.com/billy-inn/HRERE & https://github.com/billy-inn/tensorflow-efe


## Dependencies/Prerequisites:

•	tensorflow >= r1.2

•	hyperopt

•	gensim

•	sklearn

## **Description:**
Knowledge Bases (KBs) require constant updating to reflect changes to the world they represent. For general purpose KBs, this is often done through Relation Extraction (RE), the task of predicting KB relations expressed in text mentioning entities known to the KB. One way to improve RE is to use KB Embeddings (KBE) for link prediction. However, despite clear connections between RE and KBE, little has been done toward properly unifying these models systematically. HRERE helps close the gap with a framework that unifies the learning of RE and KBE models leading to significant improvements over the state-of-the-art in RE.

 ![](https://github.com/devinaarvind/ditk/blob/develop/extraction/relation/HRERE/images/hrere.png)


## **Input and Output:**
•	The **input** data for training as well as testing is in the form of a **text (.txt)** format. The **structure of the input data is FREEBASE FB15k DATA** which consists of a collection of **triplets (synset, relation_type, triplet)** extracted from Freebase (http://www.freebase.com). This data set can be seen as a 3-mode tensor depicting ternary relationships between synsets. The structure of the input data is as follows:

**/m/027rn**	/location/country/form_of_government	**/m/06cx9**

**/m/017dcd**	/tv/tv_program/regular_cast./tv/regular_tv_appearance/actor	**/m/06v8s0**

**/m/07s9rl0**	/media_common/netflix_genre/titles	**/m/0170z3**

**/m/01sl1q**	/award/award_winner/awards_won./award/award_honor/award_winner	**/m/044mz_**
 


•	The **output is an overall loss (cross-entropy loss)** to measure the dissimilarity between two distributions (language and knowledge representations).

## **Benchmark Datasets:**
•	Freebase FB15k data

•	WORDNET TENSOR DATA

## **Evaluation results:**

 ![](https://github.com/devinaarvind/ditk/blob/develop/extraction/relation/HRERE/images/hrere_metrics.png)
 

## **Models**
The generic abstract model is defined in **model.py**. All specific models are implemented in **efe.py**. The default model selected for this code is “Fb15k”.

## How to run:

_Note: The input files **(train.txt, test.txt, valid.txt)** have been added in the **config.py** as a mapping to the dataset and model being used. You will need to create a folder in the “data” folder which will be your dataset name and keep the train, test and valid files in that folder._

**To run the code, enter the command:**

python3 hrere.py

To preprocess a new dataset, ensure that it is in the format of triplets (as described above),  save the files in any folder within the “**data**” folder. Since the default dataset is set to fb15k, save the train.txt, test.txt and valid.txt in the fb15k folder and run the “**preprocess.py**” program.  

**YouTube link:**

**Note: The video shows the code being run on the Terminal.**

https://youtu.be/2wX1ds21DdI

**_Note: This code gives the output in the form of overall loss which is guided by three loss functions: one for the language representation, another for the knowledge representation, and a third one to ensure these representations do not diverge._** 
