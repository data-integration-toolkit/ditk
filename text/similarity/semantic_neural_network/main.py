import sys
from semantic_neural_network import Semantic_Neural_Network

# =============================================================
# 					Sample workflow:
# ==============================================================
def main(input_file_paths):
    dataset_name = 'Generic'
    myModel = Semantic_Neural_Network()
    read_dataset_output = myModel.read_dataset(input_file_paths, dataset_name)
    prediction_output = myModel.train(read_dataset_output)
    output_file = myModel.evaluate(read_dataset_output['test_input'], prediction_output)

    print("Results stored in "+str(output_file))

    test_sentence1_list = 'The DVD CCA then appealed to the state Supreme Court'
    text_sentence2_list = 'The DVD CCA appealed that decision to the U.S. Supreme Court'
    predictions_score = myModel.predict(test_sentence1_list, text_sentence2_list, dataset_name)

    print("Prediction score is "+ str(predictions_score))


if __name__ == '__main__':
    if(len(sys.argv) < 4):
        print("Include Train, Dev and Test file paths")
        sys.exit(-1)
    input_file_paths=[]
    input_file_paths.append(str(sys.argv[1]))
    input_file_paths.append(str(sys.argv[2]))
    input_file_paths.append(str(sys.argv[3]))
    main(input_file_paths)