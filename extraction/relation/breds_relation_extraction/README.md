# Method Name
- Semi-supervised bootstrapping of relationship extractors with distributional semantics
- David S. Batista, Bruno Martins, and M´ario J. Silva. 2015. Semi-supervised bootstrapping of relationship extractors with distributional semantics. Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing. 


## Original Code
This is the [link](https://github.com/davidsbatista/BREDS) for the original code.

## Dependencies
Besides the requirements.txt, you also need to install NLTK's Treebank PoS-tagger
```
    import nltk
    nltk.download('maxent_treebank_pos_tagger')
```

And you need download the word2vec model [afp_apw_xin_embeddings.bin](https://drive.google.com/file/d/0B0CbnDgKi0PyZHRtVS1xWlVnekE/view?usp=sharing) and put it in the same directory

## Description

This model mainly use word embedding Bootstrapping method to deal with relation extraction problem. And it is a special task since it is a semi-supervised algorithm and used for specific dataset. I tried a lot to change this one to a general model. I will explain it in the next part. Let's first look at what's in the paper.
Here's the overall model of the breds.
(img/overall_model.png)

Approach:
- Find seed matches
 1. Start with a few seed instances of a relationship type (e.g., headquartered), find text segments where they cooccur,and extract 3 contexts.

 2. Look for ReVerb relational patterns in the BETWEEN context, based on PoStags.
(img/sentence.png)
 3. Transform each context into an embedding vector with a simple compositional function that removes stopwords and adjectives, and then sums the embeddings of each word.
 (img/embedding.png)

- Generate extraction patterns(Clusters of instances)

- Find relationship instances
  - Extract segments of text with the seed's semantic types (e.g., <ORG, LOC>) and generate the embeddings context (i.e., BEF, BET, AFT).

- Handle Semantic Drift
  - Rank the extracted instances according to a confidence score, based on the patterns and similarity scores.Every instance whose confidence is above a threshold is added to the seed set and used in the next bootstrapping iteration.

## Efforts

1. I change the general input format to the required format for the model. In the original code, It needs data with format 
```
The tech company <ORG>Soundcloud</ORG> is based in <LOC>Berlin</LOC>, capital of Germany.
<ORG>Pfizer</ORG> says it has hired <ORG>Morgan Stanley</ORG> to conduct the review.
<ORG>Allianz</ORG>, based in <LOC>Munich</LOC>, said net income rose to EUR 1.32 billion.
<ORG>Pfizer</ORG>, based in <LOC>New York City</LOC> , employs about 90,000 workers.
```
So I need to change the format myself.


3. The model need positive seeds file and negative seeds file for each relation. So I write my own train function to figure out what's the relation types in the datasets and what's the entity type for each relation. Based on this, I store all relations ans entity-pairs with these relations as well as their entity types. Also since it is a bootstrapping algorithm, if the data are not large enough we can not find anything. So I filtered relations with less than 50 entity-pairs.

4. The original code can only decide one type of relation once. So I need generate positive and negative seeds file for each relation and call the origin model in a loop. It's not a easy thing since actually it is a manual work. But if I do it manually. I cannot make my model suitable for all datasets. So I must think a way to change it to automatical work. It's too hard to do. I tried different ways.
	- I use only relation type as model relation, which means if the original dataset has 5 relations, my model will also have 5.
	- I use relation type as well as entity type as model relation. For example, I will get relation (employee, PERSOM, COMPANY) and (employee, PERSON, PERSON). These are two different relationship in BREDS model. Since the seeds need point out the entity type.
	- I tried both and neither of them can give results for DDI and SemEval datasets. And for NYT dataset, it can only get a relatively low score. I dig into the code and find the original model only deal with sentences with the between words of two entities no more than 2. And NYT has 24 relationships. If I combine relations with entity type, I will get around 150 relations in all. This is one reason why the score is low. As for DDI and Semeval, they are too small to get results. 

5. Based on this case, I do some special part in my work. I get the most used relations in the dataset and pre-assign this relation to all the data. In DDI, it have a unimaginable great results.

6. The output of the model only has the sentences and entities which can be extract from using Bootstrapping. It has no corresponding thing with original data. So in order to output the general format, I store the sensences, entities, groundtruth relations in my model and output it corresponding using sentences itself as the key.

7. I write evaluation part myself since the original model can only deal with 4 relation types and evluate based on it. It use freebase, wiki, dbpedia. It's such a complex part can do not use any information from dataset. So I write my own evaluation function and since it is a general one without and special pre-process. This is the second reason why the score is low.

8. Bootstrapping is not a good method for small dataset and noisy dataset. Also it only specify the really specific dataset and relationship as well as entity type. I tried my best to change it to a general one. Now we can run it for every type of dataset and if we carefully change the parameter, filter dataset to a less noisy one and pre-assign relation, we can get some results. However, it we want to do all thing automatically, I can not guarantee any results. I only can make sure is that this model can be runned on any dataset. And pre-assigning is not part of Breds model, it's just because I have no choice.

## Input and Output
- Input for training/prediction
```
sentence e1 e1_type e1_start_pos e1_end_pos e2 e2_type e2_start_pos e2_end_pos relation (separated by tab)
```
- output for prediction
```
sentence e1 e2 predicted_relation grandtruth_relation
```

## Evalution
- Benchmark datasets
**NYT**, **DDI**,**SemEval**,**WikiData**(private dataset)
- Evaluation metrics and results
  -metrics:precision,recall,F1

If we only run the Breds model in original dataset with no process. only the results for wiki is good, since this dataset is news article(type of dataset in paper), have 12000 sentences in all and well-formatted. We can also get results from NYT because it is also news article dataset. But it is noisy so results are not good. I tried original dataset and the evlaution method from paper. It's almost the same as the paper. But it is a greatly large dataset and it is not RE group cbenchmark. I think this model only works for large dataset and 4 relations in the paper.
 (img/results_table.png)
 (img/results_picture.png)

If I pre-assign relation, choose the parameter more wisely and pre-process the data to be a less noisy one which are suiable for Breds models. I can get different results. I think this is not bad. But you need do a lot of manual work and try the best way. It's unstable and specific for evary dataset.
(img/good_results_tabls.png)
(img/good_results_picture.png)

## Demo
- Jupyter Notebook is at the same location as this README.md
- [Link](https://youtu.be/6J0I3rkYepo) to the video on Youtube
