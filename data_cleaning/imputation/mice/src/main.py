import pandas as pd
import numpy as np
import csv
from mice import Mice

def main(input_file_path):
    imputer = Mice(n_iter=5, sample_posterior=True, random_state=3)
    data_old = pd.read_csv(input_file_path,header=None)
    data=imputer.preprocess(input_file_path)  
    # data.to_csv("data_old.csv")
    n_imputations = 5
    data_complete = []
    for i in range(n_imputations):
        imputer = Mice(n_iter=5, sample_posterior=True, random_state=i)
        data_complete.append(imputer.impute(data))

    # data_complete_mean = np.round(np.mean(data_complete, 0),2)
    data_complete_mean = imputer.pool(data_complete, "mean") #pooling by mean
    data_complete_std = np.std(data_complete, 0)
    data_n=pd.DataFrame(data_complete_mean)
    output_file_path = "./input_output_generation/imputation_mice_output.csv"
    data_n.to_csv(output_file_path, index=False, header= False)
    rmse=imputer.evaluate(data_old,data_n)
    print(rmse)

    return output_file_path


if __name__ == "__main__":
    main("./input_output_generation/imputation_mice_input.csv")