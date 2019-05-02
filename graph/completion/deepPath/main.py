import os
import sys

module_path = os.path.abspath(os.path.join('../../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from graph.completion.deepPath.deep_path import DeepPath

def main (input_file_path) :
    deeppath = DeepPath()
    dataset_return = deeppath.read_dataset(input_file_path)
    deeppath.train(dataset_return)
    return deeppath.predict(dataset_return)

if __name__ == '__main__':
    main ("./data/sample_input.txt")
