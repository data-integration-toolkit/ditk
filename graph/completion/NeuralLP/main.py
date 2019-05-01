from kgc_refactor import KGC_NeuralLP

#getting the path to the project
import sys,os
project_location = os.path.dirname(__file__)
print project_location
sys.path.insert(0,project_location+"/")

def main():

	obj=KGC_NeuralLP()

	#to the read dataset function, give the path to where the datasets are saved
	#ensure that the datasets folder contains facts, train, test and valid files

	obj.read_dataset(project_location+'/datasets/kinship')
	obj.train()
	obj.predict()

	#to the get_truths function, give the path to where the datasets are saved

	obj.get_truths(project_location+'/datasets/kinship')
	obj.evaluate(obj.truths_file)
        #obj.save_model(project_location)
        #obj.load_model(project_location)
	#obj.save_model()
	#obj.load_model()

if __name__ == "__main__":
	main()

