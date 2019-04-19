# Code for Python3

# MisGAN
Data should exist in "imputation/data"
MisGAN is compatible with all three datasets

## Preprocessing:
For preprocessing training data without split run:
python main.py --preprocess

For preprocessing training data with split run:
python main.py --preprocess --split=<ratio>, where ratio is a float

## Training
For training misgan, run
python main.py --train --fname=<file>, where file name is the file name of the data loader

## Testing
For testing misgan, run
python main.py --test --fname=<fname> --model=<model>, where file name is the file name of the data loader, and model is
the name of the model before .csv
eg.

python main.py --test --fname=wdbc.csv_test --model=wdbc

## Evaluation
python main.py --evaluate --fname=<file> --model=<model>, where file is input data name and model is imputer 
model name.
eg.
python main.py --evaluate --fname="data/wdbc.csv" --model="wdbc.csv_train"

To introduce missing value manually, use:
python main.py --evaluate --fname="data/wdbc.csv" --model="wdbc.csv_train" --ims --ratio=<ratio>, ratio is float

## Imputation
python main.py --impute --fname=<file> --model=<model>, where file is input data name and model is imputer 
model name.
eg.
python main.py --impute --fname="data/wdbc.csv" --model="wdbc.csv_train"

To introduce missing value manually, use:
python main.py --evaluate --fname="data/wdbc.csv" --model="wdbc.csv_train" --ims --ratio=<ratio>, ratio is float