import os
import numpy as np

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    print(os.path.abspath("./"))
    f = open(gloveFile,'r', encoding="utf8")
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

if __name__ == "__main__":
    glove = loadGloveModel(os.path.join("glove.6B", "glove.6B.50d.txt"))
    print(glove["array"])
