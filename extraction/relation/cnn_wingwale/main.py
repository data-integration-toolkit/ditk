import os
import cnn_model

def main(input_file_path):
    input_file_path = input_file_path.replace(os.sep, '/')
    output_file_path = []

    print("Benchmark 1: DDI2013")
    model1 = cnn_model.CNNModel()
    model1.read_dataset(input_file_path+'/benchmark1')
    # model1.train(input_file_path+'/benchmark1', input_file_path+'/model1')
    output_dir1 = model1.predict(input_file_path+'/benchmark1', trained_model=input_file_path+'/model1')
    precision1, recall1, f11 = model1.evaluate(input_file_path+'/benchmark1')
    output_file_path.append(output_dir1)
    print("Precision: "+str(precision1))
    print("Recall: "+str(recall1))
    print("F1: "+str(f11))
    print()

    print("Benchmark 2: SemEval2010 Task8")
    model2 = cnn_model.CNNModel()
    model2.read_dataset(input_file_path+'/benchmark2')
    # model2.train(input_file_path+'/benchmark2', input_file_path+'/model2')
    output_dir2 = model2.predict(input_file_path+'/benchmark2', trained_model=input_file_path+'/model2')
    precision2, recall2, f12 = model2.evaluate(input_file_path+'/benchmark2')
    output_file_path.append(output_dir2)
    print("Precision: "+str(precision2))
    print("Recall: "+str(recall2))
    print("F1: "+str(f12))
    print()

    model3 = cnn_model.CNNModel()
    model3.read_dataset(input_file_path+'/benchmark3')
    # model3.train(input_file_path+'/benchmark3', input_file_path+'/model3')
    output_dir3 = model3.predict(input_file_path+'/benchmark3', trained_model=input_file_path+'/model3')
    precision3, recall3, f13 = model3.evaluate(input_file_path+'/benchmark3')
    output_file_path.append(output_dir3)
    print("Precision: "+str(precision3))
    print("Recall: "+str(recall3))
    print("F1: "+str(f13))

    return output_file_path

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main(dir_path)
