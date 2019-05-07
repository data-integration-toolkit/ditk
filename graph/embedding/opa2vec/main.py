import os
import sys

if os.name == 'nt':
    module_path = os.path.abspath(os.path.join('..\..\..'))
else:
    module_path = os.path.abspath(os.path.join('../../..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from opa2vec import OPA2VEC


def main(file_name):
    #find ditk_path from sys.path
    

    #instantiate the implemented class
    opa_obj = OPA2VEC()

    # read data
    ontology,association = opa_obj.read_dataset(file_name)


    # Learn embedding
    
    embeddings = opa_obj.learn_embeddings(file_name)

    #print(embeddings)
    #evaluation
    results = opa_obj.evaluate(embeddings)
    print(results)


if __name__ == "__main__":
    #find ditk_path from sys.path
    ditk_path = ""
    for path in sys.path:
        if "ditk" in path:
            ditk_path = path
    print(ditk_path)
    file_name = []
    onto = ditk_path+"/graph/embedding/opa2vec/test/graph_embedding_input1.owl"
    association = ditk_path+"/graph/embedding/opa2vec/test/graph_embedding_input2.txt"
    output_file = ditk_path+"/graph/embedding/opa2vec/test/sample_output.txt"
    file_name.append(onto)
    file_name.append(association)
    file_name.append(output_file)
    main(file_name)