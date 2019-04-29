# MisGAN
Data can be found in "imputation/data"
MisGAN is compatible with all three datasets

### Title of the paper
Learning from Incomplete Data with Generative Adversarial Networks

### Full citation
Li, S. C. X., Jiang, B., & Marlin, B. (2018). Learning from Incomplete Data with Generative Adversarial Networks. ICLR 2019 https://arxiv.org/abs/1902.09599
https://github.com/steveli/misgan

### Input/Output format for prediction
Input:
-Incomplete data table in csv
Output:
-Complete data table in csv

### Input/Output format for training
Input:
-Incomplete data table in csv
-Complete data table with missing rate in csv
Output:
-Complete data table in csv

### A paragraph describing the overall task, the method and model
The method uses three generator and three discriminator to train a generative adverserial network (GAN) throught
blocks of data segments in the hope to be able to regenerate complete missing data giving small fragments of real
data. 

### A figure describing the model
![alt text](https://raw.githubusercontent.com/uyuyuyjk/ditk/develop/data_cleaning/imputation/img/misgan.png)
![alt text](https://raw.githubusercontent.com/uyuyuyjk/ditk/develop/data_cleaning/imputation/img/misgan-impute.png)

### Benchmark datasets
1. UCI Letter Recognition
2. UCI Breast Cancer 
3. UCI Spambase

### Evaluation metrics and results


### Link to Jupyter notebook and Youtube videos
Jupyter Notebook:
Link1: 
Link2: 

Youtube Video:
https://www.youtube.com/watch?v=UGUKTw2Tb6k&feature=youtu.be

## Process for running the code using the main file

### Preprocessing:
For preprocessing training data without split run:
python main.py --preprocess

For preprocessing training data with split run:
python main.py --preprocess --split=<ratio>, where ratio is a float

### Training
For training misgan, run
python main.py --train --fname=<file>, where file name is the file name of the data loader
eg
python main.py --train --fname=wdbc.csv_train

### Testing
For testing misgan, run
python main.py --test --fname=<fname> --model=<model>, where file name is the file name of the data loader, and model is
the name of the model before .csv
eg.

python main.py --test --fname=wdbc.csv_test --model=wdbc

### Evaluation
python main.py --evaluate --fname=<file> --model=<model>, where file is input data name and model is imputer 
model name.
eg.
python main.py --evaluate --fname="data/wdbc.csv" --model="wdbc.csv_train"

To introduce missing value manually, use:
python main.py --evaluate --fname="data/wdbc.csv" --model="wdbc.csv_train" --ims --ratio=<ratio>, ratio is float

### Imputation
python main.py --impute --fname=<file> --model=<model>, where file is input data name and model is imputer 
model name.
eg.
python main.py --impute --fname="data/wdbc.csv" --model="wdbc.csv_train"

To introduce missing value manually, use:
python main.py --evaluate --fname="data/wdbc.csv" --model="wdbc.csv_train" --ims --ratio=<ratio>, ratio is float