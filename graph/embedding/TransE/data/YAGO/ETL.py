import pandas as pd
import numpy as np

def ETL_train(file_name,output_name,rel_dict, entity_d):

	train_data = pd.read_csv(file_name, header=None, delim_whitespace=True)
	train_data = train_data.loc[:,0:2]
	train_data.columns = ["A","C","B"]
	#columnsTitles=["A","C","B"]
	
	train_data["C"] = train_data["C"].astype(int)
	train_data = train_data.replace({"C": rel_dict})
	
	train_data["A"] = train_data["A"].astype(int)
	train_data = train_data.replace({"A": entity_d})
	
	train_data["B"] = train_data["B"].astype(int)
	train_data = train_data.replace({"B": entity_d})
	
	train_data=train_data[["A","B","C"]]
	#print(train_data)
	train_data.to_csv(output_name, sep='\t', index=False, header=False)

def entity_2_id(file_name,output_name):
	entityid = pd.read_csv(file_name, header=None, delim_whitespace=True)
	entityid = entityid.loc[:,0:1]
	entityid.to_csv(output_name, sep='\t', index=False, header=False)
	
if __name__ == '__main__':
	relation_d = {}
	with open("./relation2id.txt") as f:
		for line in f:
		   (entity, id) = line.split()
		   relation_d[int(id)] = entity
	
	entity_d = {}
	with open("./entity2id.txt", encoding="utf8") as f:
		for line in f:
		   (entity, id) = line.split()
		   entity_d[int(id)] = entity

	ETL_train('./train.txt','./new_train.txt',relation_d, entity_d)
	ETL_train('./test.txt','./new_test.txt',relation_d, entity_d)
	ETL_train('./valid.txt','./new_valid.txt',relation_d, entity_d)
	
	#entity_2_id('./entity2id.txt','./new_entity2id.txt')
	
