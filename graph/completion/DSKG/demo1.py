from dskg import *


obj=Dskg()
obj.read_dataset('train',"Freebase")
obj.train()
obj.predict()

mrr,f1=obj.evaluate()

print("MRR= ",mrr)
print("F1= ",f1)

entity=open('/Users/anshuman786/PycharmProjects/CSCI548-Project/entity2id').readlines()
relations=open('/Users/anshuman786/PycharmProjects/CSCI548-Project/relation2id').readlines()



for i in range(4):
    print("Prediction= ",obj.data[i][0]," ",obj.data[i][1]," ", obj.predicted_entity[i])
    print("Actual= ", obj.data[i][0], " ", obj.data[i][1], " ", obj.data[i][2])

    prediction=None
    actual=None
    subject=None
    relation=None



    for elements in entity:
        e,index=elements.split()

        if(index==str(obj.predicted_entity[i])):
            prediction=e
            break

    for elements in entity:
        e, index = elements.split()
        if (index == str(obj.data[i][2])):
            actual = e
            break

    for elements in entity:
        e, index = elements.split()
        if (index == str(obj.data[i][0])):
            subject = e
            break

    for elements in relations:
        e, index = elements.split()
        if (index == str(obj.data[i][1])):
            relation = e
            break

    print("Mapping1= ",subject," ",relation," ",prediction)
    print("Mapping2= ",subject," ",relation," ",actual)
    print("")


