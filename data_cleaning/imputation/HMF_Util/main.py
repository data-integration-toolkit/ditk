from nmf_np_ref import HMF_Class

def main():
	obj=HMF_Class()

	#for 'inputData', give the path where the datasets are saved

	inputData = "datasets/"
	fileName = "wdbc.csv"
	R,M,K,init_UV,iterations=obj.preprocess(inputData=inputData, fileName = fileName)
	obj.impute()

if __name__=='__main__':
    main()
