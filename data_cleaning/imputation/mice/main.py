import pandas as pd
import numpy as np
import csv
from Mice import mice

imputer = mice(n_iter=5, sample_posterior=True, random_state=3)
data_old = pd.read_csv("letter-recognition.csv",header=None)
data=imputer.preprocess("letter-recognition.csv")  
# data.to_csv("data_old.csv")
n_imputations = 5
data_complete = []
for i in range(n_imputations):
    imputer = mice(n_iter=5, sample_posterior=True, random_state=i)
    data_complete.append(imputer.impute(data))

data_complete_mean = np.round(np.mean(data_complete, 0),2)
data_complete_std = np.std(data_complete, 0)
data_n=pd.DataFrame(data_complete_mean)
data_n.to_csv("result.csv", index=False, header= False)
rmse=imputer.evaluate(data_old,data_n)
print(rmse)
