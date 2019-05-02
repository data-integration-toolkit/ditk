import os
import cnn_model

def main(input_file_path):
    input_file_path = input_file_path.replace(os.sep, '/')

    print("Benchmark 3: NYT dataset")
    model3 = cnn_model.CNNModel()
    model3.read_dataset(input_file_path+'/benchmark3')
    output_file_path = model3.predict(input_file_path+'/benchmark3', trained_model=input_file_path+'/model3')
    precision3, recall3, f13 = model3.evaluate(input_file_path+'/benchmark3')
    print("Precision: "+str(precision3))
    print("Recall: "+str(recall3))
    print("F1: "+str(f13))

    return output_file_path

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main(dir_path)
