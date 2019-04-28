import os 
from ditk.misc.rescal_wingwale.rescal_model import RESCALModel
import sys

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    print("Benchmark 1: Alyawarra kinship data")
    model1 = RESCALModel()
    model1.read_dataset(dir_path+'\\rescal_input_alyawarradata.mat', 'Rs')
    A1, R1 = model1.factorize(50)

    mean_train1, std_train1, mean_test1, std_test1 = model1.evaluate()
    print("PR AUC training/test mean: %.3f/%.3f" %(mean_train1, mean_test1))
    print("PR AUC training/test standard deviation: %.3f/%.3f" %(std_train1, std_test1))
    print()

    print("Benchmark 2: UMLS dataset")
    model2 = RESCALModel()
    model2.read_dataset(dir_path+'\\rescal_input_umls.mat', 'Rs')
    A2, R2 = model2.factorize(50)

    mean_train2, std_train2, mean_test2, std_test2 = model2.evaluate()
    print("PR AUC training/test mean: %.3f/%.3f" %(mean_train2, mean_test2))
    print("PR AUC training/test standard deviation: %.3f/%.3f" %(std_train2, std_test2))
    print()

    print("Benchmark 3: human disease-symptoms data resource")
    model3 = RESCALModel()
    model3.read_dataset(dir_path+'\\rescal_input_diseases.mat', 'K')
    A3, R3 = model3.factorize(50)

    mean_train3, std_train3, mean_test3, std_test3 = model3.evaluate()
    print("PR AUC training/test mean: %.3f/%.3f" %(mean_train3, mean_test3))
    print("PR AUC training/test standard deviation: %.3f/%.3f" %(std_train3, std_test3))
    print()
    