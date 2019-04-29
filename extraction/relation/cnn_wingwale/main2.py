import os
import cnn_model

def main(input_file_path):
    input_file_path = input_file_path.replace(os.sep, '/')

    print("Benchmark 2: SemEval2010 Task8")
    model2 = cnn_model.CNNModel()
    model2.read_dataset(input_file_path+'/benchmark2')
    # model2.train(input_file_path+'/benchmark2', input_file_path+'/model2')
    output_file_path = model2.predict(input_file_path+'/benchmark2', trained_model=input_file_path+'/model2')
    precision2, recall2, f12 = model2.evaluate(input_file_path+'/benchmark2')
    print("Precision: "+str(precision2))
    print("Recall: "+str(recall2))
    print("F1: "+str(f12))

    return output_file_path

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main(dir_path)
