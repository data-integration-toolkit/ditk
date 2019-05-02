#!/usr/bin/env python
# coding: utf-8

# In[1]:


from urllib.request import urlretrieve # Python 3
import os
import h5py
from annoy import AnnoyIndex
import pandas as pd
import csv
from ParentAPI import Blocking

# Import the parent abstract method file
class Annoy(Blocking): 
    def __init__(self,**kargs): 
        self.labels = kargs['labels'] if 'labels' in kargs else []
        self.trainData = kargs['train_data'] if 'train_data' in kargs else []
        self.testData = kargs['test_data'] if 'test_data' in kargs else []
        self.distanceMetric = kargs['metric'] if 'metric' in kargs else 'angular'
        self.numTrees = kargs['n_trees'] if 'n_trees' in kargs else 10
        self.numResults = kargs['num_results'] if 'num_results' in kargs else 10
        self.searchK = kargs['search_k'] if 'search_k' in kargs else -1
        self.includeDistance =  kargs['include_distances'] if 'include_distances' in kargs else False
        self.vectorLength = kargs['vector_length'] if 'vector_length' in kargs else None
        
    def read_dataset(self,filepath_list='glove-25-angular',*argv,**kargs):
        if type(filepath_list) == str: 
            filepath_list = [filepath_list]
        datasets = []
        for path_or_url in filepath_list: 
            if os.path.exists(path_or_url): 
                dataset = pd.read_csv(open(path_or_url,'r'))
                datasets.append(dataset)
            else: 
                if not os.path.exists(path_or_url+'.hdf5'):  
                    print('Dataset not found, downloading from http://vectors.erikbern.com')
                    urlretrieve('http://vectors.erikbern.com/%s.hdf5'%path_or_url, path_or_url+'.hdf5')
                else: 
                    print('Found '+path_or_url+'.hdf5')
                hdfFile = h5py.File(path_or_url+'.hdf5','r')
                train = pd.DataFrame(hdfFile.get('train').value)
                self.trainData = train
                datasets.append(train)
                test = pd.DataFrame(hdfFile.get('test').value)
                self.testData = test
                datasets.append(test)
                groundTruth = pd.DataFrame(hdfFile.get('neighbors').value)
                self.labels = groundTruth
                datasets.append(groundTruth)
                self.distanceMetric = hdfFile.attrs['distance'] if self.distanceMetric == None else 'angular'
            
        if len(datasets) == 1: 
            datasets = datasets[0]
        if 'isTrain' in kargs and kargs['isTrain']: 
            self.trainData = datasets
        if 'isTest' in kargs and kargs['isTest']: 
            self.testData = datasets
        if 'isLabels' in kargs and kargs['isLabels']: 
            self.labels = datasets
        return datasets
    
    def train(self,dataframe=None,*argv,**kargs): 
        # Dataframe index used as Annoy index when building trees
        distanceMetric = kargs['metric'] if 'metric' in kargs else self.distanceMetric
        numTrees = kargs['n_trees'] if 'n_trees' in kargs else self.numTrees
        dataset = dataframe if dataframe is not None else self.trainData
        
        if len(dataset.columns) > 1: 
            data = [x[1].values for x in dataset.iterrows()]
        else: 
            data = [x[1][0] for x in dataset.iterrows()]

        vectorLength = kargs['vector_length'] if 'vector_length' in kargs else [len(data[0]) if hasattr(data[0],'len') else data[0].shape[0]][0]
        self.vectorLength = vectorLength
        annoy = AnnoyIndex(vectorLength,distanceMetric)
        for idx,vector in zip(dataset.index,data): 
            annoy.add_item(idx,vector)
        annoy.build(numTrees)
        self.annoy = annoy
        return annoy
    
    def predict(self,dataframe_list=None,*argv,**kargs):
        if dataframe_list is not None and type(dataframe_list) is not list: 
            dataframe_list = [dataframe_list]
        annoy = self.annoy
        dataset = dataframe_list[0] if dataframe_list is not None else self.testData
        numResults = kargs['num_results'] if 'num_results' in kargs else self.numResults
        searchK = kargs['search_k'] if 'search_k' in kargs else self.searchK
        includeDistance =  kargs['include_distances'] if 'include_distances' in kargs else self.includeDistance
        
        if len(dataset.columns) > 1: 
            data = [x[1].values for x in dataset.iterrows()]
        else: 
            data = [x[1][0] for x in dataset.iterrows()]
        
        results = []
        if (type(data[0]) is list and len(data[0]) > 1) or (hasattr(data[0],'shape') and data[0].shape[0] > 1): 
            for vector in data: 
                results.append(annoy.get_nns_by_vector(vector,numResults,searchK,includeDistance))
        else:
            for index in data: 
                results.append(annoy.get_nns_by_item(index,numResults,searchK,includeDistance))

        return results
        
    def evaluate(self,groundtruth=None,dataframe_list=None,*argv,**kargs): 
        if dataframe_list is not None and type(dataframe_list) is not list: 
            dataframe_list = [dataframe_list]
        annoy = self.annoy
        dataset = dataframe_list[0] if dataframe_list is not None else self.testData
        groundTruthDF = groundtruth if groundtruth is not None else self.labels
        searchK = kargs['search_k'] if 'search_k' in kargs else self.searchK
        includeDistance =  kargs['include_distances'] if 'include_distances' in kargs else self.includeDistance

        if len(dataset.columns) > 1: 
            data = [x[1].values for x in dataset.iterrows()]
        else: 
            data = [x[1][0] for x in dataset.iterrows()]
            
        if len(groundTruthDF.columns) > 1: 
            labels = [x[1].values for x in groundTruthDF.iterrows()]
        else: 
            labels = [x[1][0] for x in groundTruthDF.iterrows()]
        numResults = len(labels[0]) if hasattr(labels[0],'len') else labels[0].shape[0]
        numResults = kargs['num_results'] if 'num_results' in kargs else numResults
        
        truePositives = 0; falsePositives = 0; falseNegatives = 0; 
        if (type(data[0]) is list and len(data[0]) > 1) or (hasattr(data[0],'shape') and data[0].shape[0] > 1): 
             for vector,labels in zip(data,labels):
                predictions = annoy.get_nns_by_vector(vector,numResults,searchK,includeDistance)
                predictions = predictions[0] if self.includeDistance else predictions
                numCorrect = sum([1 for idx,prediction in enumerate(predictions) if prediction in labels])
                truePositives += numCorrect
                falsePositives += len(labels) - numCorrect
                falseNegatives += len(labels) - numCorrect
                reductionRatio = numResults / [labels.shape[0] if hasattr(labels,'shape') else len(labels)][0]
        else: 
            for index,labels in zip(data,labels):
                predictions = annoy.get_nns_by_item(index,numResults,searchK,includeDistance)
                predictions = predictions[0] if self.includeDistance else predictions
                numCorrect = sum([1 for idx,prediction in enumerate(predictions) if prediction in labels])
                truePositives += numCorrect
                falsePositives += len(labels) - numCorrect
                falseNegatives += len(labels) - numCorrect
        reductionRatio = numResults / len(dataset.index)
        precision = truePositives/(truePositives+falsePositives)
        recall = truePositives/(truePositives+falseNegatives)        
        return precision, recall, reductionRatio
    
    def save_model(self,path):
        if type(path) != str: 
            print('Path must be a string.')
        self.annoy.save(path)
        
    def load_model(self,path,metric=None,vector_length=None):
        metric = metric if metric is not None else self.distanceMetric
        vectorLength = vector_length if vector_length else self.vectorLength
        try: 
            annoy = AnnoyIndex(vectorLength,metric)
            annoy.load(path)
            self.annoy = annoy 
        except: 
            print('Metric and vector length must match those used to build the Annoy Index you are loading.')
    


# In[ ]:




