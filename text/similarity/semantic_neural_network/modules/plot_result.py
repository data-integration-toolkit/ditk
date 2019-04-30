#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
import pickle

from modules.configs import PRETRAIN, RESULTS_DIR


def plot_fit_history(history, model_file, title):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    if title:
        plt.title(title.decode('utf-8'))
    plt.ylabel('Custo')
    plt.xlabel('Iteração'.decode('utf-8'))
    plt.legend(['Treino', 'Validação'.decode('utf-8')], loc='upper left')
    filename = os.path.join(RESULTS_DIR, model_file + '.svg')
    plt.savefig(filename)

def save_history(history, model_file):
    filename = os.path.join(RESULTS_DIR, model_file + '.history.p')
    with open(filename, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
