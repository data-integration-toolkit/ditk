#!/usr/bin/env python3
import model


def main(input):
    # Initializing Model and Set Parameters
    m = model.local_global_CRNN_bio(inputPath=input)

    # Read Raw Dataset
    m.read_dataset()

    # Split into Train: 90% Test: 10
    # (No Dev because we aren't playing with Hyper Params)
    m.data_preprocess()

    # Parse with our group's Common Format and
    m.tokenize()

    # Train with our model
    m.train()

    # Generate Prediction File
    outputPath = m.predict()

    # Evaluate with group's metric
    m.evaluate()

    # Return Prediction File Path
    print('Path to Prediction File: ' + outputPath)
    return outputPath


if __name__ == "__main__":
    main('/home/ashgu/Desktop/548ProjectFinal/ditk/extraction/relation/local_global_crnn/Data/SemEval/SemEval.txt')
