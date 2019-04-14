# Template module for Python version 3.x

# MisGAN

## loading the data
Data should exist in "/data"
MisGAN will be compatible with all three datasets
python misgan_main.py --preprocess

## Training
For training misgan 
python misgan_main.py --fname=<fname> --train --misgan

For training misgan imputer
python misgan_main.py --fname=<fname> --train --imputer

Fmputer is trained after misgan finishes training, for simplicity hyper parameter for training are predefined

## Testing
For testing misgan 
python misgan_main.py --model=<model_name> --fname=<fname> --test --misgan

For testing misgan imputer
python misgan_main.py  --model=<model_name> --fname=<fname> --test --imputer

## Evaluation
Evaluation of MisGAN uses RMSE score
python misgan_main.py --model=<model_name> <fname>



# new commands
Preprocess:
python misgan_main.py --preprocess --split=0.8

Evaluation

python misgan_main.py --evaluate --fname="data/wdbc.csv" --model="wdbc_imputer.pth"