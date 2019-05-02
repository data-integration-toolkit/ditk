#Python 3.x
import numpy as np
from numpy import genfromtxt
from fancyimpute import KNN
import os, sys
from os.path import isfile, join

sys.path.append("..")
from imputation import Imputation

class knnImpute(Imputation):

    def preprocess(self, inputData):
        """
	    Reads a dataset (complete dataset without missing values) and introduces missingness in the dataset.
        :param inputData: 
			FilePath to the (complete) dataset
        :return:
            X_incomplete: numpy array with dropped entries
        """
        X = genfromtxt(inputData, delimiter=',')
        # X is a data matrix which we're going to randomly drop entries from
        missing_mask = np.random.rand(*X.shape) < 0.1
        X_incomplete = X.copy()
        # missing entries indicated with NaN
        X_incomplete[missing_mask] = np.nan
        return X_incomplete


    def train(self, train_data):
        # KNN is a lazy learning machine learning algorithm - no training is required
        pass


    def test(self, trained_model, test_data):
        # No testing
        pass
		
    def impute(self, trained_model, input):
        """
        Loads the input table and gives the imputed table
    
    	:param trained_model: trained model returned by train function - not used in our case
    	:param input: input table which needs to be imputed
    	:return:
    		X_filled_knn: imputed table as a numpy array
        """
        # Use 3 nearest rows which have a feature to fill in each row's missing features
        # will not use trained_model as training happens during imputation
        X_incomplete = input
        knnImpute = KNN(k=3)
        X_filled_knn = knnImpute.fit_transform(X_incomplete)
        return X_filled_knn
    

    def evaluate(self, trained_model, input, *args, **kwargs):
        """
        Loads the original dataset and calculates the performance on the imputed table through RMSE.

        :param trained_model: trained model returned by train function- not used in our case
        :param input: imputed table on which model needs to be evaluated
        :param kwargs:
            kwargs.inputData: FilePath to the (complete) dataset
        :return:
            knn_mse: rmse
        """
        inputData = kwargs['inputData']      
        X_filled_knn = input
        X = genfromtxt(inputData, delimiter=',')
        missing_mask = np.random.rand(*X.shape) < 0.1
        #take X, original table through args
        knn_mse = ((X_filled_knn[missing_mask] - X[missing_mask]) ** 2).mean()
        # normalize the RMSE
        knn_mse = knn_mse if knn_mse < 1 else knn_mse / 1000
        return knn_mse
    
    def save_model(self, file):
        # No models saved
        pass

    def load_model(self, file):
        # No models loaded
        pass
        
def main(input_file_path) : 
	# your processing code
    # switch to data directory which is outside our codebase and contains the dataset
    input_file_path = join(os.pardir, "data", input_file_path)
    
    #print a numpy array without scientific notation
    np.set_printoptions(suppress=True)
    
    # create an instance of the knnImpute class
    knnimpute = knnImpute()

    # preprocess the data - introduce missingness
    preprocess = knnimpute.preprocess(input_file_path)
    print("Incomplete Data:")
    print(preprocess)

    # impute the data
    impute = knnimpute.impute(trained_model = '', input = preprocess)
    print("Imputed Data:")
    print(impute)

    # evaluate the imputed data with RMS Error
    evaluate = knnimpute.evaluate(trained_model = '', input = impute, inputData = input_file_path)
    print("RMSE:")
    print(evaluate)

    # save imputed data as a csv
    output_file_path = "imputation_test_output.csv"
    np.savetxt("imputation_test_output.csv", impute, delimiter=",")
    
    # return file path of the imputed data (stored as a csv file)
    return output_file_path

if __name__ == "__main__":
    # input to the code is input file path i.e. file path of data to be imputed
    # output is file path of imputed table (as a csv file)
    main('wdbc.csv')