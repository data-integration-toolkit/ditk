import sys
from Global_Normalization import Global_Normalization

# =============================================================
# 					Sample workflow:
# ==============================================================
def main(input_file_paths):
    myModel = Global_Normalization()
    dataset_name = 'CoNLL04'
    config_file = 'configs/config.crf.setup1'
    setupOption = 1
    read_dataset_output_path, test_id2sent, test_id2arg2rel = myModel.read_dataset(input_file_paths, dataset_name, config_file, setupOption)
    output_model_file, evaluation_result, output_file = myModel.train(read_dataset_output_path,dataset_name,config_file, test_id2sent, test_id2arg2rel)

    print("Output Model file is at location "+ output_model_file)
    print("F1 score is "+str(evaluation_result))
    print("Output stored in "+ output_file)

if __name__ == '__main__':
    if(len(sys.argv) < 3):
        print("Include Train and Test file paths")
        sys.exit(-1)
    input_file_paths=[]
    input_file_paths.append(str(sys.argv[1]))
    input_file_paths.append(str(sys.argv[2]))
    main(input_file_paths)