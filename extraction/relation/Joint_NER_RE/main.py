import multhead_joint_entity_relation_extraction
from multhead_joint_entity_relation_extraction import multihead_joint_entity_relation_extraction
obj=multihead_joint_entity_relation_extraction()
filename="/home/sanjitha/Downloads/relation_extraction_test_input.txt"
data=obj.read_dataset(filename)

obj.data_preprocess(data)
obj.train()
obj.evaluate()
