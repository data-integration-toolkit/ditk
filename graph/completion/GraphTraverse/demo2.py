from graphTraverse import *

dataset_path="/Users/anshuman786/PycharmProjects/CSCI548-Project/"
params_path="/Users/anshuman786/PycharmProjects/CSCI548-Project/SINGLE.CPKL"
glove_path="/Users/anshuman786/PycharmProjects/CSCI548-Project/glove.6B.50d.txt"

obj=GraphTraverse()

obj.read_dataset([dataset_path,params_path,glove_path],"FreeBase")

obj.train()

obj.evaluate()


data=obj.dset.test

for i in range(4):
    val=obj.predict(data[i])
    if(val==True):
        print(data[i]," is True")
    else:
        print(data[i], "is False")
