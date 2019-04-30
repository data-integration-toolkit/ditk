from keras.callbacks import Callback
import prepare_data as prd
import configs
import numpy as np
from seqeval.metrics import f1_score, classification_report
from keras.callbacks import ModelCheckpoint
import model as mdl
from paper1_modified import End_to_end as ner

# if we want to go through the whole process
def main():
    filenames = [configs.TRAINING_FILE, configs.VALIDATION_FILE]
    data1 = ner.read_dataset(filenames)
    ner.train(data1)
    data2 = ner.predict()
    precision, recall, f1Score = ner.evaluation(data2[0])
    print(precision, recall, f1Score)
 #   return data2

## if we already have a trained h5 file and we don't want to spend much time training again, use this function
def main2():
    data2 = ner.predict()
    precision, recall, f1Score = ner.evaluation(data2[0])
    print(precision, recall, f1Score)
#    return data2

if __name__ == '__main__':
    main()