#!/usr/bin/env python
# coding: utf-8

# In[32]:


from urllib.request import urlretrieve # Python 3
import os
import h5py
import nmslib 
import pandas as pd
from ParentAPI import Blocking

# Import the parent abstract method file
class NMSLIB(Blocking): 
    def __init__(self,**kargs): 
        self.labels = kargs['labels'] if 'labels' in kargs else []
        self.trainData = kargs['train_data'] if 'train_data' in kargs else []
        self.testData = kargs['test_data'] if 'test_data' in kargs else []
        self.distanceMetric = kargs['metric'] if 'metric' in kargs else 'l2'
        self.m = kargs['M'] if 'M' in kargs else 15
        self.numResults = kargs['num_results'] if 'num_results' in kargs else 10
        self.efc = kargs['efc'] if 'efc' in kargs else 100
        self.numThreads =  kargs['indexThreadQty'] if 'indexThreadQty' in kargs else 4
        self.includeDistances = kargs['include_distances'] if 'include_distances' in kargs else False
        
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
                self.distanceMetric = hdfFile.attrs['distance'] if self.distanceMetric == None else 'l2'
                if self.distanceMetric == 'angular': 
                    self.distanceMetric = 'angulardis'
            
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
        distanceMetric = kargs['metric'] if 'metric' in kargs else self.distanceMetric
        dataset = dataframe if dataframe is not None else self.trainData
        M = kargs['M'] if 'M' in kargs else self.m
        efc = kargs['efc'] if 'efc' in kargs else self.efc
        numThreads = kargs['indexThreadQty'] if 'indexThreadQty' in kargs else self.numThreads
        
        if len(dataset.columns) > 1: 
            data = [x[1].values for x in dataset.iterrows()]
        else: 
            data = [x[1][0] for x in dataset.iterrows()]
        index = nmslib.init(method='hnsw', space=distanceMetric,data_type=nmslib.DataType.DENSE_VECTOR)
        index.addDataPointBatch(data)
        index.createIndex({'M': M,'indexThreadQty':numThreads,'efConstruction':efc}, print_progress=True)
        self.numSamples = len(data)
        
        self.index = index
        return index
    
    def predict(self,dataframe_list=None,*argv,**kargs):
        index = self.index
        if dataframe_list is not None and type(dataframe_list) is not list: 
            dataframe_list = [dataframe_list]
        dataset = dataframe_list[0] if dataframe_list is not None else self.testData
        numResults = kargs['num_results'] if 'num_results' in kargs else self.numResults
        numThreads = kargs['num_threads'] if 'num_threads' in kargs else self.numThreads
        efs = kargs['efSearch'] if 'efSearch' in kargs else self.efc
        includeDistances = kargs['inlude_distances'] if 'include_distances' in kargs else self.includeDistances
        
        index.setQueryTimeParams({'efSearch':efs})
        if len(dataset.columns) > 1: 
            query = [x[1].values for x in dataset.iterrows()]
        else: 
            query = [x[1][0] for x in dataset.iterrows()]
        if includeDistances: 
            results = index.knnQueryBatch(query, k=numResults, num_threads=numThreads)
        else: 
            results = [x[0] for x in index.knnQueryBatch(query, k=numResults, num_threads=numThreads)]
        return results
        
    def evaluate(self,groundtruth=None,dataframe_list=None,*argv,**kargs): 
        index = self.index
        if dataframe_list is not None and type(dataframe_list) is not list: 
            dataframe_list = [dataframe_list]
        test = dataframe_list[0] if dataframe_list is not None else self.testData
        groundTruth = groundtruth if groundtruth is not None else self.labels
        
        numThreads = kargs['num_threads'] if 'num_threads' in kargs else self.numThreads
        efs = kargs['efSearch'] if 'efSearch' in kargs else self.efc
                
        if len(test.columns) > 1: 
            query = [x[1].values for x in test.iterrows()]
        else: 
            query = [x[1][0] for x in test.iterrows()]
        
        if len(groundTruth.columns) > 1: 
            labels = [x[1].values for x in groundTruth.iterrows()]
        else: 
            labels = [x[1][0] for x in groundTruth.iterrows()]
        numResults = len(labels[0]) if hasattr(labels[0],'len') else labels[0].shape[0]
        numResults = kargs['num_results'] if 'num_results' in kargs else numResults
        truePositives = 0; falsePositives = 0; falseNegatives = 0; 
        predictions = [x[0] for x in index.knnQueryBatch(query, k=numResults, num_threads=numThreads)]
        
        numCorrect = sum([1 for idx,prediction in enumerate(predictions) if prediction in labels[idx]])
        truePositives += numCorrect
        falsePositives += len(labels) - numCorrect
        falseNegatives += len(labels) - numCorrect
        reductionRatio = numResults / self.numSamples

        reductionRatio = numResults / len(test.index)
        precision = truePositives/(truePositives+falsePositives)
        recall = truePositives/(truePositives+falseNegatives)        
        return precision, recall, reductionRatio
    
    def save_model(self,path):
        if type(path) != str: 
            print('Path must be a string.')
        self.index.saveIndex(path)
        
    def load_model(self,path,metric=None):
        metric = metric if metric is not None else self.distanceMetric
        try: 
            newIndex = nmslib.init(method='hnsw', space=metric) 
            newIndex.loadIndex(path)
            self.index = newIndex 
        except: 
            print('Metric must match those used to build the index you are loading.')
    
    


# In[ ]:





# In[ ]:




