MCDSAT - Scalable Query Rewriting using Materialized Views
----------------------------------------------------------

Requirements
------------
Python 3 or newer and the following libraries:

---------------------------------------------------------------------------------------------
|   PACKAGE NAME                                            |     INSTALLATION              |
---------------------------------------------------------------------------------------------
|   TPG -> http://christophe.delord.free.fr/en/tpg/         |   python3 setup.py install    |
|   (Included)                                              |                               |
---------------------------------------------------------------------------------------------
|   c2d -> Darwiche's c2d compiler                          |   Set the right path while    |
|   http://reasoning.cs.ucla.edu/c2d/ (Included)            |   calling read_input in mcdsat|
---------------------------------------------------------------------------------------------
|   dnnf-models -> for model enumeration (Included)         |   Uncompress the tar          |
|   For more details on usage, check the README file        |   cd nnf; make                |
|   in the dnnf-models folder                               |   make                        |
---------------------------------------------------------------------------------------------
|   psyco -> http://psyco.sourceforge.net/                  |                               |
|   (not included, only needed for improved performace)     |                               |
|   if used, must uncomment the psyco lines in Main.py      |                               |
---------------------------------------------------------------------------------------------

Installation
------------
Download this folder and follow the above instructions to fulfill the necessary package requirements
Keep queries and views in the right folders and pass along the right paths in the jupyter notebook.

Introduction
------------
This is a Python3 Implementation of the paper - 

Compilation of Query-Rewriting Problems into Tractable Fragments of Propositional Logic
Yolife Arvelo, Blai Bonet and Maria Esther Vidal, 
Proc. 21st National Conf. on Artificial Intelligence (AAAI). Boston, MA. 2006. 
AAAI Press. Pages 225-230.

Objective - Given a query Q, retrieve all tuples obtainable from the data sources that satisfy Q
---------   A rewriting is a query-like expression that refers only to the views

Data Sources are assumed to be - 
1) Independent (i.e maintained in a distributed manner)
2) Described as Views (i.e Local As View model)
3) Incomplete

Assumption
----------
- Views may be incomplete
- Query and the views are conjunctive queries and do not contain any arithmetic predicates

MCDSAT Algoritm
---------------
- Given a Query Q and a set of Views V
- Build a Propositional Theory such that its models are in correspondence with the MCDs
- Generating MCDs is now a problem of Model Enumeration
- Model Enumeration can be done with modern SAT techniques that implement :
    1) Non-chronological backtracking via clause learning
    2) Caching of common subproblems
    3) Heuristics
- Extend the propositional theory such that its models are in correspondence with the rewritings
- This approach is called McdSat

Negation Normal Forms (NNF)
---------------------------
- A formula is in NNF if constructed from literals using only conjunctions and disjunctions
- It can be represented as a rooted DAG whose leaves are literals and internal nodels are labeled with conjunction or disjunctions

Deterministic and Decomposable NNFs (d-DNNFs)
- Introduced by Darwiche
- NNF is decomposable if each variable appears at most once below each conjunct
- NNF is deterministic if disjuncts are pairwise logically inconsistent
- A d-DNNf supports a number of operations in linear time - 
    1) Satisfiability
    2) Clause entailment
    3) Model Counting
    4) Model Enumerationg (output linear time)

Implementation
--------------
- MCDSAT translates Query Rewriting Problem (QRP) into a Propostional Theory T
- T is compiled into d-DNNF using Darwiche's c2d compiler (off-the shelf component)
- Models are obtained from d-DNNF using the dnnf-models (off-the shelf component)
- These models can be transformed into MCDs or rewritings
- MCDSAT is written in Python3.5

Usage
-----
Download the mcdsat folder and follow the installation instructions for the prerequisites
After the installation is complete, the following command can be used to get the query
rewritings for the specified files

python3 main.py 

This execution generates the Query Rewritings for the
Query file examples/query_0.txt
Views file examples/views_0.txt

It accesses c2d compiler from the "c2d/c2d_linux" path
It accesses models from the "dnnf-models/models" path

These values can be changed to point to the paths which has the query and views as required.
Custom queries and views can be given as input and rewritings can be generated for them using
this module.

The output of this file is a set of rewritings generated for the aforementioned Query
using the aforementioned views file.

More details on how to use this module can be found in the Jupyter Notebook that comes with this folder.
    