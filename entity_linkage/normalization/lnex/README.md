# Location Name Extraction
## Paper Title
- Location Name Extraction from Targeted Text Streams using Gazetteer-based Statistical Language Models
##Full Citation
- Location Name Extraction from Targeted Text Streams using Gazetteer-based Statistical Language Models - Hussein S. AlOlimat, Krishnaprasad Thirunarayan, Valerie Shalin, and Amit Sheth. 2018. In Proceedings of the 27th International
Conference on Computational Linguistics (COLING 2018) 

## Original Code
https://github.com/halolimat/LNEx 

## Requirements
- Python3
- ElasticSearch on OSM (the details on installation are below)
 * Note ElasticSearch is vital to query the Gazetteer, please make sure the instance is running before testing the code

## Installing OpenStreetMap Gazetteers
We will be using a ready to go elastic index of the whole [OpenStreetMap](http://www.osm.org) data (~ 108 GB) provided by [komoot](http://www.komoot.de) as part of their [photon](https://photon.komoot.de/) open source geocoder ([project repo](https://github.com/komoot/photon)). Follow the steps below to get Photon running in your system:

 - Download the full photon elastic index which is going to allow us to query OSM using a bounding box

   ```sh
   wget -O - http://download1.graphhopper.com/public/photon-db-latest.tar.bz2 | pbzip2 -cd | tar x
   ```

 - Now, start photon which starts the elastic index in the background as a service

   ```sh
   wget https://github.com/komoot/photon/releases/download/0.3.0/photon-0.3.0.jar
   java -jar photon-0.3.0.jar
   ```

 - You can now test the running index by running the following command (9200 is the default port number, might be different in your system if the port is occupied by another application):
   ```sh
   curl -XGET 'http://localhost:9200/photon/'

## Description
- The overall task of this project is to extract location from streaming text services such as Twitter and using a Gazetteer normalize these entities and provide the lat,long of the normalized location.
- The appraoch for the task is as follows:
	- Create a bounding box for Gazetteer location : This is done for candidate generation while at the same time ensuring the search indexing speed is not a bottleneck
	- Gazetteer augmentation and filtering: This is done for creating new normalized entities and remove the entities not present in bounding box
	- Tweet Preprocessing : Standard operation of removing stopwords, using dictionary to fix spelling errors and hashtag segmentation
	- Tweet Tokenization and building a bottom up tree, a cartesian product of token is combined to form a location and then the location is checked if it is valid or not by querying the gazetteer.
	- Finally the location is extracted and normalized entity is given as output
## Functions
- The read_dataset function receives list of input file and returns the test and eval set
- The predict function extracts the tweet, preprocesses the tweet, filters the gazetteer and then creates a bottom up tree by tokenizing the tweets and checks if location is valid
- The evaluate function evaluate the predictions made by the predict function, for evaluation the input is eval set which has hand labelled ground truth. The evaluation metrics are precision,recall and F1 score


## Input and Output
- The inputs tken by the module is tweet for e.g
Sample:
```
#ChennaiFloods A pregnant lady rescued from Mudichur Near tambaram around 1130 this morning by Indian Navy https://t.co/I5ZYe4
```

- Output: A list of normalized geo-entities along with boundaries of the location and meta-data linking entity to gazetteer for lat,long
Sample:
```
tambaram, (58, 66), tambaram, ['1904', '5520']
Chennai, (1, 8), chennai, ['15543', '3197']
```

## Evalution
### Benchmark datasets 
- Twitter Data: The algorithm is benchmarked on 7500 tweets collected from Chennai, Houston and Louisiana Floods. These tweets were hand labelled to obtain the ground truth

### Evaluation metrics
- Precision
- Recall
- F1 score

Benchmark     | Precision         |   Recall|  F1     
------------- | ----------------- |-------- |------- |
Chennai       |       0.94        |  0.83   | 0.89   |
Louisiana     |       0.89        |  0.82   | 0.85   | 
Houston       |       0.73        | 0.65    | 0.73   | 


## Demo
- [Sample notebook](./main_nb.ipynb)
- [Demo Video](https://youtu.be/puFjQ_ImAvU)