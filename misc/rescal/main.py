import os 
from rescal_model import RESCALModel

def main(input_file_path):
    output_file_path = []

    print("Benchmark 1: Alyawarra kinship data")
    model1 = RESCALModel()
    model1.read_dataset(input_file_path+'\\benchmark1\\rescal_input_alyawarradata.mat', 'Rs')
    model1.factorize(50)
    output_dir1 = input_file_path+'\\benchmark1'
    model1.save_model(output_dir1)
    output_file_path.append(output_dir1)

    print("Benchmark 2: UMLS dataset")
    model2 = RESCALModel()
    model2.read_dataset(input_file_path+'\\benchmark2\\rescal_input_umls.mat', 'Rs')
    model2.factorize(50)
    output_dir2 = input_file_path+'\\benchmark2'
    model2.save_model(output_dir2)
    output_file_path.append(output_dir2)


    print("Benchmark 3: human disease-symptoms data resource")
    model3 = RESCALModel()
    model3.read_dataset(input_file_path+'\\benchmark3\\rescal_input_diseases.mat', 'K')
    model3.factorize(50)
    output_dir3 = input_file_path+'\\benchmark3'
    model3.save_model(output_dir3)
    output_file_path.append(output_dir3)

    return output_file_path

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main(dir_path)
    