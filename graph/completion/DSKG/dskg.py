#import graph_completion
from reader import *
from model import *
import tensorflow as tf



class Dskg():

    def __init__(self):
        self.usage_class = None
        self.options = Options()
        self.trainer = None
        self.right=0
        self.wrong=0
        self.predicted_entity=None
        self.data=None

    def makeentityid(self, fileName):
        file=open(fileName).readlines()
        entity=set()
        for element in file:
            s,r,o=element.split()
            entity.add(s)
            entity.add(o)

        newFile=open('entity2id','w')
        counter=0
        for element in entity:
            entry=str(element)+'\t'+str(counter)+'\n'
            newFile.write(entry)
            counter=counter+1

    def makerelationid(self, fileName):
        file = open(fileName).readlines()
        relation = set()
        for element in file:
            s, r, o = element.split()
            relation.add(r)


        newFile = open('relation2id', 'w')
        counter = 0
        for element in relation:
            entry = str(element) + '\t' + str(counter) + '\n'
            newFile.write(entry)
            counter = counter + 1



    def read_dataset(self, fileName, options):
        """
        Reads dataset from the benchmark files.
        dataset will be of the "subject-relation-object". Subject and Object are the entities.
        Converts the data into numerical format.
        Preprocesses the data to add some negative examples.
        Form entity emdeddings from subject and object and relation emdeddings from relations.
        Splits the data into train, dev and test sets.

        """
        self.makerelationid(fileName)
        self.makeentityid(fileName)

        if(options=="FreeBase"):
            self.usage_class= type('new',(Model,FreeBaseReader),dict())
        else:
            self.usage_class = type('new', (Model, WordNetReader), dict())







    def train(self):
        """
         Using the emdeddings obtained, deep learning is done using recurrent neural networks.
         dev data is used to make the model more robust
        """
        self.usage_class=type('newTest', (RespectiveTester, self.usage_class), dict())
        self.usage_class = type('newTrain', (RespectiveTrainer, self.usage_class, Printer), dict())
        session=tf.Session()
        self.trainer=self.usage_class(self.options,session)
        self.trainer.train()



    def predict(self):

        """
         from the test data we predict object on the basis of subject- relation.
         for every subject-relation embeddings a group of entities(objects) are predicted.
        """
        result1, result2=self.trainer.predict()

        predictions=result1.tolist()
        self.predicted_entity=predictions
        data=result2.tolist()
        self.data=data


        for i in range(0,len(predictions)):
            predict=predictions[i]
            s,r,o=data[i]
            if(data[i][2]==predict):
                self.right=self.right+1
            else:
                self.wrong=self.wrong+1


            '''
            entitylist=open("entity2id").readlines()
            for line in entitylist:
                line=line.split()
                if(line[1]==str(predict)):
                    print("Prediction= ",str(s)+" "+str(o)+" "+str(line[0]))
                    print("Actual= ",data[i])
            '''



    def evaluate(self, metrics={}, options={}):
        """
        Evaluates the predicted results using evaluation metrics.
        """
        mrr=(self.right*1.0)/self.wrong
        f1=(2*self.right*1.0)/(2*self.right+self.wrong)
        return mrr, f1


