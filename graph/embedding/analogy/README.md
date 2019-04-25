# ANALOGY

### Graph Embedding Project for CSCI548@USC
This code base implements the following paper:
##### Hanxiao Liu, Yuexin Wu, Yiming Yang. Analogical Inference for Multi-relational Embeddings. Proceedings of the 34th International Conference on Machine Learning, Sydney, Australia, PMLR 70, 2017. Paper: https://arxiv.org/pdf/1705.02426.pdf
##### This code base simplifies/refactors this GitHub repo: https://github.com/mana-ysh/knowledge-graph-embeddings to conform to a class project
---
#### Datasets:
This code has been tested on the FB15k and WN18 datasets. 

Note: To conform to standards for the class project I modified the relations and entities input files, different from original code. If you wish to conform to original codes relation and entity files (suggested), substitute commented code in dataset.py with uncommented (line 65 with line 71) to not read the entity and relation number

Entities Example:
```text
/m/06rf7	0
/m/0c94fn	1
/m/016ywr	2
/m/01yjl	3
```
Relations Example:
```text
/sports/sports_team/roster./soccer/football_roster_position/player	8
/business/company_type/companies_of_this_type	9
/tv/tv_program/regular_cast./tv/regular_tv_appearance/character	10
/architecture/structure/address./location/mailing_address/citytown	11
```
Train/Validate/Test Example:
```text
/m/017dcd	/m/06v8s0	/tv/tv_program/regular_cast./tv/regular_tv_appearance/actor
/m/07s9rl0	/m/0170z3	/media_common/netflix_genre/titles
/m/01sl1q	/m/044mz_	/award/award_winner/awards_won./award/award_honor/award_winner
/m/0cnk2q	/m/02nzb8	/soccer/football_team/current_roster./sports/sports_team_roster/position
```
---
#### Running the code 
#####(please load the Jupyter Notebook sample *analogy_notebook.ipynb* to aid you in running the code for the first time):
1. Initialize a model
```python
algorithm = ANALOGY()
```
2. Load the Data Set
```python
train_file_names = {"train": input_file_path + "train.txt",
# Optional              "valid": input_file_path + "valid.txt",
# Optional              "whole": input_file_path + "whole.txt",
                        "relations": input_file_path + "relation2id.txt",
                        "entities": input_file_path + "entity2id.txt"}

algorithm.read_dataset(train_file_names)
```
3. Learn the embeddings (i.e train model) *Note:* you can pass empty dictionary, there are default parameters for everything
```python
parameters = {"mode": 'single',
              "epoch": 3,
              "batch": 128,
              "lr": 0.05,
              "dim": 200,            
              "negative": 3,         
              "opt": 'adagrad',
              "l2_reg": 0.001,
              "gradclip": 5,
              'filtered': True}
              
# Other parameters not listed:
# margin
# cp_ratio
# metric
# nbest
# batch
# save_step

algorithm.learn_embeddings(parameters)
```
4. Save the model (Use if you want to save training model for use later). Some models will take days to train!
```python
algorithm.save_model(input_file_path + "analogy.mod")
```
5. Load a model (Use if you saved a model above, allows you to have to not rerun training of model)
```python
algorithm.load_model(input_file_path + "analogy.mod")
```
6. Retrieve embeddings.
```python
test_subs = ['/m/07z1m', '/m/03gqgt3', '/m/01c9f2']
print(algorithm.retrieve_entity_embeddings(test_subs))
```
7. Retrieve scoring matrix (each row is the s,r scored against each other entity)
```python
    sm = algorithm.retrieve_scoring_matrix(test_subs, test_rels)
    print(sm)
```
---
#### Evaluation Results
#####WN18
| Source        | MRR (filtered) | MRR (raw) | Hits@1 (Filtered) | Hits@3 (Filtered)
| ------------- |:--------:|:------:|:------:|:------:| 
| Paper      | 94.2 | 65.7 | 93.9 | 94.4
| Code      | 94.2 | 58.5 | 93.8 | 94.5
##### FB15k
| Source        | MRR (filtered) | MRR (raw) | Hits@1 (Filtered) | Hits@3 (Filtered)
| ------------- |:--------:|:------:|:------:|:------:| 
| Paper      | 94.1 | 58.7 | 93.6 | 94.5
| Code      | 94.3 | 58.2 | 94.0 | 94.6

