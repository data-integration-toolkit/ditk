import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from text.similarity.USE_Transformer.src.USE_Transformer import USE_Transformer_Similarity
from decimal import Decimal

def main(input_file_path):
    local_module_path = '../model/moduleA'
    remote_module_path = 'https://tfhub.dev/google/universal-sentence-encoder-large/3'
    # create an instance
    trans_ins = USE_Transformer_Similarity()
    # read_dataset
    trans_ins.read_dataset(input_file_path)
    # load_model ..use local path or remote path
    trans_ins.load_model(local_module_path)
    # predict
    sim_scores = trans_ins.predict(trans_ins.sentences_1,trans_ins.sentences_2)
    # save predict similarity scores
    file= open("../data/output_sim.txt",'w')
    for score in sim_scores:
        score = str(Decimal(score).quantize(Decimal('0.00')))
        file.write(score)
        file.write('\n')


if __name__ =="__main__":

    main('../data/input.txt')


