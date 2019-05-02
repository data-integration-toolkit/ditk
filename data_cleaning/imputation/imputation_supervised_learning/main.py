from imputation_sm import Imputation_with_supervised_learning


def main(filename):
    """
        # Sample workflow:

        inputFiles = ['thisDir/file1.txt','thatDir/file2.txt','./file1.txt']

        myModel = Imputation_with_supervised_learning()  # instantiate the class

        data_X, train_data, test_data = myModel.read_dataset(inputFiles)  # read in complete dataset and introduce missing values

        imputed_values = myModel.preprocess(data_X) # imputes values using different imputation methods
        rmse = myModel.evaluate(inputData, imputed_values) # calculate RMSE
        print('RMSE: %s'%(rmse)) #RMSE is the evaluation metric used
        
        model_train_data = myModel.train(train_data)  # Preprocessing of training data
        model_test_data = myModel.test(test_data)  # Preprocessing of testing data
        error_rate = myModel.predict(train_data, test_data)  # Error rate is used for comparison between different classifiers, imputation methods and perturbations

        """
    categorical_values = True
    dataname = 'breast_cancer'
    missing_values = True
    header = False
    if categorical_values:
        obj = Imputation_with_supervised_learning()
        input_data = obj.preprocess(filename, header, missing_values, categorical_values)
        output_filename = obj.impute(input_data, dataname, True)
        train_data, labels_train, test_data, labels_test = obj.train(input_data,dataname)
        obj.test(test_data, labels_test, dataname)
        obj.predict(dataname)
        obj.evaluate(filename)
        
    else:
        obj = Imputation_with_supervised_learning()
        input_data = obj.preprocess(filename, header, missing_values, categorical_values)
        output_filename = obj.impute(input_data, dataname)
    return output_filename
    


if __name__ == '__main__':
    filename = 'data/datasets/breast_cancer.csv'
    main(filename)
