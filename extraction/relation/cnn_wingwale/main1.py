import os
import cnn_model

def main(input_file_path):
    input_file_path = input_file_path.replace(os.sep, '/')

    print("Benchmark 1: DDI2013")
    model1 = cnn_model.CNNModel()
    model1.read_dataset(input_file_path+'/benchmark1')
    # model1.train(input_file_path+'/benchmark1', input_file_path+'/model1')
    output_file_path = model1.predict(input_file_path+'/benchmark1', trained_model=input_file_path+'/model1')
    precision1, recall1, f11 = model1.evaluate(input_file_path+'/benchmark1')
    print("Precision: "+str(precision1))
    print("Recall: "+str(recall1))
    print("F1: "+str(f11))

    return output_file_path

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main(dir_path)
